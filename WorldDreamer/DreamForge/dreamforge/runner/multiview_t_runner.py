import logging
import os
import contextlib
from functools import partial
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from dreamforge.dataset.utils import collate_fn
from ..misc.common import load_module, convert_outputs_to_fp16
from .multiview_runner import MultiviewRunner
from .base_t_validator import BaseTValidator
from .utils import smart_param_count, prepare_ckpt
from ..networks.unet_2d_condition_multiview_s import UNet2DConditionModelMultiviewScene
from ..networks.mutual_self_attention import ReferenceTransformerControl


class ControlnetUnetEmbedderWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, scene_embedder, can_bus_embedder, reference_control_reader, weight_dtype=torch.float32,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.scene_embedder = scene_embedder
        self.can_bus_embedder = can_bus_embedder
        self.reference_control_reader = reference_control_reader
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(self, noisy_latents, noisy_latents_unet, scene_embedding, can_bus, timesteps, timesteps_unet, camera_param, encoder_hidden_states,
                encoder_hidden_states_uncond, controlnet_image, **batch_kwargs):
        N_cam = noisy_latents.shape[1]
        # fmt: off
        down_block_res_samples, mid_block_res_sample, \
        encoder_hidden_states_with_cam = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            camera_param=camera_param,  # b, N_cam, 189
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
            controlnet_cond=controlnet_image,  # b, 26, 200, 200
            return_dict=False,
            scene_embedding=scene_embedding,
            **batch_kwargs,
        )
        # fmt: on
        # starting from here, we use (B n) as batch_size
        noisy_latents_unet = rearrange(noisy_latents_unet, "b n ... -> (b n) ...")
        if timesteps_unet.ndim == 1:
            timesteps_unet = repeat(timesteps_unet, "b -> (b n)", n=N_cam)
        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        # Predict the noise residual
        
        can_bus_embedding = self.can_bus_embedder(can_bus.float())[:, None]
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            scene_embedding = [rearrange(scene_embedding, "b n ... -> (b n) ...").to(dtype=self.weight_dtype)]
            self.reference_control_reader.update(scene_embedding +[can_bus_embedding.to(dtype=self.weight_dtype)])
            model_pred = self.unet(
                noisy_latents_unet,  # b x n, 4, H/8, W/8
                timesteps_unet.reshape(-1),  # b x nc
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample
            # self.reference_control_reader.clear()  # Important !!!

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)

        return model_pred


class MultiviewTRunner(MultiviewRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = BaseTValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
            }
        )
        # we set _sc_attn_index here
        if cfg.model.sc_attn_index:
            self._sc_attn_index = OmegaConf.to_container(
                cfg.model.sc_attn_index, resolve=True)
        else:
            self._sc_attn_index = None

    def get_sc_attn_index(self):
        return self._sc_attn_index

    def _set_dataset_loader(self):
        # dataset
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "foreground_loss_mode": self.foreground_loss_mode,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
            "keyframe_rate": self.cfg.runner.keyframe_rate,
            "ref_length": self.cfg.model.ref_length
        }

        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                collate_fn=partial(
                    collate_fn, is_train=True, **collate_fn_param),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers, pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=True,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                collate_fn=partial(
                    collate_fn, is_train=False, **collate_fn_param),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        assert cfg.model.pretrained_dreamforge is not None

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        if cfg.model.fix_controlnet:
            unet = UNet2DConditionModelMultiviewScene.from_pretrained(
                cfg.model.pretrained_dreamforge, subfolder=cfg.model.unet_dir)
            self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)
        else:
            self.unet = model_cls.from_pretrained(
                cfg.model.pretrained_dreamforge, subfolder=cfg.model.unet_dir, **unet_param)

        self.reference_control_reader = ReferenceTransformerControl(self.unet, fusion_blocks="full")

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True)
        controlnet = model_cls.from_pretrained(
            cfg.model.pretrained_dreamforge, subfolder=cfg.model.controlnet_dir)
        self.controlnet = model_cls.from_unet(controlnet, load_param=True, **controlnet_param)  # Note that we change that to support load param to a model with new parameters

        model_cls = load_module(cfg.model.scene_embedder_cls)
        scene_embedder_param = OmegaConf.to_container(
            self.cfg.model.scene_embedder, resolve=True)
        self.scene_embedder = model_cls(**scene_embedder_param)
        self.accelerator.wait_for_everyone()
        state_dict = torch.load(os.path.join(cfg.model.pretrained_dreamforge, cfg.model.scene_embedder_dir, "scene_embedder_model.bin"), map_location='cpu')
        self.scene_embedder.load_param(state_dict)
        logging.info(
                f"[MultiviewRunner] load weight of scene_embedder")

        model_cls = load_module(cfg.model.can_bus_embedder_cls)
        can_bus_embedder_param = OmegaConf.to_container(
            self.cfg.model.can_bus_embedder, resolve=True)
        self.can_bus_embedder = model_cls(**can_bus_embedder_param)
        if not cfg.model.fix_controlnet:
            self.accelerator.wait_for_everyone()
            state_dict = torch.load(os.path.join(cfg.model.pretrained_dreamforge, cfg.model.can_bus_embedder_dir, "can_bus_embedder_model.bin"), map_location='cpu')
            self.can_bus_embedder.load_state_dict(state_dict)
            logging.info(
                    f"[MultiviewRunner] load weight of can_bus_embedder")

        # add setter func
        for mod in self.unet.modules():
            if hasattr(mod, "_sc_attn_index"):
                mod._sc_attn_index = self.get_sc_attn_index

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # controlnet 
        if self.cfg.model.fix_controlnet:
            self.controlnet.requires_grad_(False)
            self.controlnet.train(False)
            # scene embedder
            for name, mod in self.scene_embedder.named_parameters():
                mod.requires_grad_(False)
        else:
            self.controlnet.train(train)
            # scene embedder
            for name, mod in self.scene_embedder.named_parameters():
                logging.debug(
                    f"[MultiviewRunner] set {name} to requires_grad = True")
                mod.requires_grad_(train)
        # unet
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            print(mod)
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)
        # can_bus embedder
        for name, mod in self.can_bus_embedder.named_parameters():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = [] if self.cfg.model.fix_controlnet else list(self.controlnet.parameters())
        if not self.cfg.model.fix_controlnet:
            scene_embedder_params = list(self.scene_embedder.parameters())
            params_to_optimize += scene_embedder_params
            logging.info(
                f"[MultiviewRunner] add {smart_param_count(scene_embedder_params)} params from scene_embedder to optimizer.")
        can_bus_embedder_params = list(self.can_bus_embedder.parameters())
        logging.info(
            f"[MultiviewRunner] add {smart_param_count(can_bus_embedder_params)} params from can_bus_embedder to optimizer.")
        params_to_optimize += can_bus_embedder_params
        unet_params = self.unet.trainable_parameters
        logging.info(
            f"[MultiviewRunner] add {smart_param_count(unet_params)} params from unet to optimizer.")
        params_to_optimize += unet_params
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        self.controlnet_unet_embedder = ControlnetUnetEmbedderWrapper(self.controlnet, self.unet, self.scene_embedder, self.can_bus_embedder, self.reference_control_reader)
        # accelerator
        ddp_modules = (
            self.controlnet_unet_embedder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet_embedder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.accelerator.unwrap_model(
                        self.unet).trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")

        controlnet_unet_embedder = self.accelerator.unwrap_model(self.controlnet_unet_embedder)
        controlnet_unet_embedder.weight_dtype = self.weight_dtype
        controlnet_unet_embedder.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(
                self.cfg,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder
            )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        if not self.cfg.model.fix_controlnet:
            controlnet = self.accelerator.unwrap_model(self.controlnet)
            controlnet.save_pretrained(os.path.join(root, self.cfg.model.controlnet_dir))
            logging.info(f"Save controlnet model to: {root}")

            scene_embedder = self.accelerator.unwrap_model(self.scene_embedder)
            os.makedirs(os.path.join(root, self.cfg.model.scene_embedder_dir), exist_ok=True)
            torch.save(scene_embedder.state_dict(), os.path.join(root, self.cfg.model.scene_embedder_dir, 'scene_embedder_model.bin'))
            logging.info(f"Save scene_embedder model to: {root}")

        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save unet model to: {root}")

        can_bus_embedder = self.accelerator.unwrap_model(self.can_bus_embedder)
        os.makedirs(os.path.join(root, self.cfg.model.can_bus_embedder_dir), exist_ok=True)
        torch.save(can_bus_embedder.state_dict(), os.path.join(root, self.cfg.model.can_bus_embedder_dir, 'can_bus_embedder_model.bin'))
        logging.info(f"Save can_bus_embedder model to: {root}")

    def _train_one_step(self, batch):
        self.controlnet_unet_embedder.train()
        with self.accelerator.accumulate(self.controlnet_unet_embedder):
            N_frame = batch["pixel_values"].shape[1]
            N_cam = batch["pixel_values"].shape[2]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(
                    batch["pixel_values"],
                    "b l n c h w -> (b l n) c h w").to(
                    dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(
                latents, "(b l n) c h w -> b l n c h w", l=N_frame, n=N_cam)
            
            scene_embedding, _ = self.scene_embedder(latents, batch['meta_data'])

            # TODO: ref image latent encoding shape_like text encoder
            ref_latents = self.vae.encode(
                rearrange(
                    batch["ref_values"],
                    "b l n c h w -> (b l n) c h w").to(
                    dtype=self.weight_dtype)).latent_dist.sample()
            ref_latents = ref_latents * self.vae.config.scaling_factor
            ref_latents = rearrange(
                ref_latents, "(b l n) c h w -> b l n c h w", l=self.cfg.model.ref_length, n=N_cam)

            ref_can_bus = batch["ref_can_bus"].to(self.weight_dtype)
            can_bus = batch["can_bus"].to(self.weight_dtype)
            can_bus = torch.cat([ref_can_bus, can_bus], dim=1)
            can_bus = rearrange(can_bus, "b l c -> (b l) c")
            can_bus = repeat(can_bus, 'b ... -> (b n) ...', n=N_cam)     

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, :, 1], "b l ... -> b l r ...", r=N_cam)
            if self.cfg.model.train_with_same_noise_t:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_frame)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            assert self.cfg.model.train_with_same_t
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            # add frame dim
            timesteps_unet = repeat(timesteps, "b ... -> b r ...", r=N_frame+self.cfg.model.ref_length)
            timesteps_unet = timesteps_unet.long()
            timesteps = timesteps_unet[:, self.cfg.model.ref_length:]

            #### here we change (b, l, ...) to (bxl, ...) ####
            f_to_b = partial(rearrange, pattern="b l ... -> (b l) ...")
            b_to_f = partial(
                rearrange, pattern="(b l) ... -> b l ...", l=N_frame)
            latents = f_to_b(latents)
            noise = f_to_b(noise)
            timesteps = f_to_b(timesteps)
            timesteps_unet = f_to_b(timesteps_unet)
            camera_param = f_to_b(camera_param)
            if batch['kwargs']['bboxes_3d_data'] is not None:
                batch_kwargs = {
                    "bboxes_3d_data": {
                        'bboxes': f_to_b(batch['kwargs']['bboxes_3d_data']['bboxes']),
                        'classes': f_to_b(batch['kwargs']['bboxes_3d_data']['classes']),
                        'masks': f_to_b(batch['kwargs']['bboxes_3d_data']['masks']),
                    }
                }
            else:
                batch_kwargs = {"bboxes_3d_data": None}

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)
            noisy_latents_unet = f_to_b(torch.cat([ref_latents, b_to_f(noisy_latents)], dim=1))

            #### here we change (b, l, ...) to (bxl, ...) ####
            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                f_to_b(batch["input_ids"]))[0]
            encoder_hidden_states_uncond = self.text_encoder(
                f_to_b(batch["uncond_ids"]))[0]
            
            bev_hdmap = batch["bev_hdmap"].to(dtype=self.weight_dtype)
            bev_hdmap = f_to_b(bev_hdmap)
            layout_canvas = batch["layout_canvas"].to(dtype=self.weight_dtype)
            layout_canvas = f_to_b(layout_canvas)
            controlnet_image = [bev_hdmap, layout_canvas]

            model_pred = self.controlnet_unet_embedder(noisy_latents, noisy_latents_unet, scene_embedding, can_bus, timesteps, timesteps_unet, camera_param, encoder_hidden_states,
                encoder_hidden_states_uncond, controlnet_image, **batch_kwargs)

            #### change dims back ####
            noise = b_to_f(noise)
            model_pred = b_to_f(model_pred)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.cfg.runner.max_grad_norm is not None:
                params_to_clip = self.controlnet_unet_embedder.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)
            self.reference_control_reader.clear()  # Important !!!

        return loss

    def _validation(self, step):
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        unet = self.accelerator.unwrap_model(self.unet)
        can_bus_embedder = self.accelerator.unwrap_model(self.can_bus_embedder)
        scene_embedder = self.accelerator.unwrap_model(self.scene_embedder)
        image_logs = self.validator.validate(
            controlnet, unet, scene_embedder, can_bus_embedder, self.accelerator.trackers, step,
            self.weight_dtype, self.accelerator.device)