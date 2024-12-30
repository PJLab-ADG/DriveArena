import logging
import os
import contextlib
from omegaconf import OmegaConf
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler

from .base_runner import BaseRunner
from .utils import smart_param_count
from dataset.utils import collate_fn_singleframe
from projects.dreamer.utils.common import load_module, convert_outputs_to_fp16, move_to
from projects.dreamer.runner.single_ref_validator import SingleRefBaseValidator
from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2


class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(
        self, controlnet, unet, weight_dtype=torch.float32, unet_in_fp16=True
    ) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(
        self,
        noisy_latents,
        timesteps,
        camera_param,
        encoder_hidden_states,
        encoder_hidden_states_uncond,
        bev_hdmap,
        rel_pose,
        ref_images,
        layout_canvas,
        camera_params_raw=None,
        **kwargs,
    ):
        N_cam = noisy_latents.shape[1]
        kwargs = move_to(kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)

        # fmt: off
        down_block_res_samples, mid_block_res_sample, \
        encoder_hidden_states_with_cam = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            camera_param=camera_param,  # b, N_cam, 189
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
            bev_hdmap=bev_hdmap,  # b, 3, 200, 200
            rel_pose=rel_pose, 
            ref_images=ref_images,
            layout_canvas=layout_canvas,
            return_dict=False,
            # camera_params_raw=camera_params_raw,
            **kwargs,
        )
        # fmt: on

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_with_cam.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                # camera_params_raw=camera_params_raw,
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred


class MultiviewSingleRefRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = SingleRefBaseValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "embedder": self.embedder,
            },
        )

    def _init_fixed_models(self, cfg):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="vae"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.embedder = FrozenOpenCLIPImageEmbedderV2(
            arch="ViT-B-32",
            model_path="./pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
        )

    def _init_trainable_models(self, cfg):
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="unet"
        )

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True
        )
        self.controlnet = model_cls.from_unet(unet, **controlnet_param)

        self.image_proj_model = nn.Linear(
            cfg.model.image_proj_model.input_dim, cfg.model.image_proj_model.output_dim
        )

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)
        self.unet.requires_grad_(False)
        self.image_proj_model.requires_grad_(True)

        self.embedder.train = False
        for param in self.embedder.parameters():
            param.requires_grad = False

        for name, mod in self.unet.trainable_module.items():
            logging.debug(f"[MultiviewRunner] set {name} to requires_grad = True")
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
        params_to_optimize = list(self.controlnet.parameters())
        unet_params = self.unet.trainable_parameters
        proj_params = self.image_proj_model.parameters()
        param_count = smart_param_count(unet_params) + smart_param_count(proj_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet and proj to optimizer."
        )

        params_to_optimize += unet_params
        params_to_optimize += proj_params

        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps
            * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps
            * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )

    def _set_dataset_loader(self):
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
            "with_ref_bboxes": self.cfg.runner.with_ref_bboxes,
        }
        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=True,
                collate_fn=partial(
                    collate_fn_singleframe, is_train=True, **collate_fn_param
                ),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers,
                pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=self.cfg.runner.persistent_workers,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                shuffle=False,
                collate_fn=partial(
                    collate_fn_singleframe, is_train=False, **collate_fn_param
                ),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def prepare_device(self):
        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)
        # accelerator
        ddp_modules = (
            self.controlnet_unet,
            self.image_proj_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet,
            self.image_proj_model,
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
        self.embedder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.image_proj_model.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.unet.trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(dtype=torch.float16)(
                        mod.forward
                    )
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32."
                )
        controlnet_unet = self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype = self.weight_dtype
        controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16
        self.accelerator.unwrap_model(self.embedder)
        self.accelerator.unwrap_model(self.image_proj_model)

        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(
                self.cfg, tokenizer=self.tokenizer, text_encoder=self.text_encoder
            )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(os.path.join(root, self.cfg.model.controlnet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        image_proj_model = self.accelerator.unwrap_model(self.image_proj_model)
        os.makedirs(os.path.join(root, self.cfg.model.image_proj_model_dir), exist_ok=True)
        torch.save(
            image_proj_model.state_dict(),
            os.path.join(
                root, self.cfg.model.image_proj_model_dir, "image_proj_model.bin"
            ),
        )
        logging.info(f"Save your model to: {root}")

    def _train_one_step(self, batch):
        self.controlnet_unet.train()
        with self.accelerator.accumulate(self.controlnet_unet):
            N_cam = batch["pixel_values"].shape[1]

            # Convert ref_images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack(
                    [
                        torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        for _ in range(N_cam)
                    ],
                    dim=1,
                )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(batch["uncond_ids"])[0]

            bev_hdmap = batch["bev_hdmap"].to(dtype=self.weight_dtype)
            camera_param = batch["camera_param"].to(self.weight_dtype)
            rel_pose = batch["relative_pose"].to(self.weight_dtype)

            layout_canvas = batch["layout_canvas"].to(dtype=self.weight_dtype)

            if hasattr(self, "embedder"):
                image_tensor = rearrange(
                    batch["ref_images"], "b n c h w -> (b n) c h w"
                ).to(
                    dtype=self.weight_dtype
                )  # torch.Size([6, 3, 224, 400])
                # img: b c h w >> b l c
                ref_images = self.embedder(image_tensor)  # torch.Size([6, 50, 768])
                ref_images = self.image_proj_model(ref_images)
                ref_images = rearrange(ref_images, "(b n) c l -> b n c l", n=N_cam)

            model_pred = self.controlnet_unet(
                noisy_latents,
                timesteps,
                camera_param,
                encoder_hidden_states,
                encoder_hidden_states_uncond,
                bev_hdmap,
                rel_pose,
                ref_images,
                layout_canvas,
                **batch["kwargs"],
            )

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean()

            self.accelerator.backward(loss)
            if (
                self.accelerator.sync_gradients
                and self.cfg.runner.max_grad_norm is not None
            ):
                params_to_clip = self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.cfg.runner.set_grads_to_none)

        return loss

    def _validation(self, step):
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        unet = self.accelerator.unwrap_model(self.unet)
        image_proj_model = self.accelerator.unwrap_model(self.image_proj_model)
        image_logs = self.validator.validate(
            controlnet,
            unet,
            self.accelerator.trackers,
            step,
            self.weight_dtype,
            self.accelerator.device,
            image_proj_model,
        )
