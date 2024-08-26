from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect

import torch
import PIL
import numpy as np
from einops import rearrange

from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import BaseOutput, randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ..utils.common import move_to


class StableDiffusionSingleRefControlNetPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        controlnet,
        scheduler: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        embedder = None,
        image_proj_model = None,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        assert safety_checker == None, "Please do not use safety_checker."
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_resize=False,
            do_convert_rgb=False,
            do_normalize=False,
        )
        self.embedder = embedder
        self.image_proj_model = image_proj_model

    def numpy_to_pil_double(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        We need to handle 5-dim inputs and reture 2-dim list.
        """
        imgs_list = []
        for imgs in images:
            imgs_list.append(self.numpy_to_pil(imgs))
        return imgs_list

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            raise RuntimeError("If you fixed the logic for generator, please remove this. Otherwise, please use other sampler.")
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        # decode latents with 5-dims
        latents = 1 / self.vae.config.scaling_factor * latents

        bs = len(latents)
        latents = rearrange(latents, 'b c ... -> (b c) ...')
        image = self.vae.decode(latents).sample
        image = rearrange(image, '(b c) ... -> b c ...', b=bs)

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = rearrange(image.cpu(), '... c h w -> ... h w c').float().numpy()
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        bev_hdmap: torch.FloatTensor,
        camera_param: Union[torch.Tensor, None],
        rel_pose: torch.FloatTensor,    # b, 4, 4
        ref_images: torch.FloatTensor,    # b, 6, 3, 224, 400
        layout_canvas: torch.FloatTensor,    # b, 6, 13, 224, 400
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1,
        guess_mode: bool = False,
        use_zero_map_as_unconditional: bool = False,
        bev_controlnet_kwargs = {},
        camera_params_raw = None,
        bbox_max_length = None,
        use_ref_img_in_noise=False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        # BEV: we cannot use the size of image
        # height, width = self._default_height_width(height, width, None)

        # 1. Check inputs. Raise error if not correct
        # we do not need this, only some type assertion
        # self.check_inputs(
        #     prompt,
        #     image,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        # )

        # 2. Define call parameters
        # NOTE: we get batch_size first from prompt, then align with it.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        ### BEV, check camera_param ###
        if camera_param is None:
            # use uncond_cam and disable classifier free guidance
            N_cam = 6  # TODO: hard-coded
            camera_param = self.controlnet.uncond_cam_param((batch_size, N_cam))
            do_classifier_free_guidance = False
        ### done ###

        # if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 3. Encode input prompt
        # NOTE: here they use padding to 77, is this necessary?
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )  # (2 * b, 77 + 1, 768)

        # 4. Prepare image
        # NOTE: if image is not tensor, there will be several process.
        assert not self.control_image_processor.config.do_normalize, "Your controlnet should not normalize the control image."
        image = self.prepare_image(
            image=bev_hdmap,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )  # (2 * b, c_26, 200, 200)
        if use_zero_map_as_unconditional and do_classifier_free_guidance:
            # uncond in the front, cond in the tail
            _images = list(torch.chunk(image, 2))
            _images[0] = torch.zeros_like(_images[0])
            image = torch.cat(_images)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if not use_ref_img_in_noise:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,  # will use if not None, otherwise will generate
            )  # (b, c, h/8, w/8) -> (bs, 4, 28, 50)
        else:
            latent_timesteps = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            latents = self.prepare_latents_with_ref_img(
                rearrange(ref_images, "b n c h w -> (b n) c h w"),
                latent_timesteps,
                batch_size,
                prompt_embeds.dtype,
                device,
                generator,
            )
            

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ###### BEV: here we reconstruct each input format ######
        assert camera_param.shape[0] == batch_size, \
            f"Except {batch_size} camera params, but you have bs={len(camera_param)}"
        N_cam = camera_param.shape[1]
        if not use_ref_img_in_noise:
            latents = torch.stack([latents] * N_cam, dim=1)  # bs, 6, 4, 28, 50
        else:
            latents = rearrange(latents, "(b n) c h w -> b n c h w", b=batch_size)
        # prompt_embeds, no need for b, len, 768
        # image, no need for b, c, 200, 200
        camera_param = camera_param.to(self.device)
        rel_pose = move_to(rel_pose, self.device)
        ref_images =  move_to(ref_images, self.device)
        layout_canvas =  move_to(layout_canvas, self.device)

        if do_classifier_free_guidance and not guess_mode:
            # if camera_params_raw is not None:
            #     camera_params_raw = {
            #         k: torch.cat([v] * 2) for k, v in camera_params_raw.items()
            #     }
            # uncond in the front, cond in the tail
            _images = list(torch.chunk(image, 2))
            kwargs_with_uncond = self.controlnet.add_uncond_to_kwargs(
                camera_param=camera_param,
                image=_images[0],  # 0 is for unconditional
                max_len=bbox_max_length,
                **bev_controlnet_kwargs,
                rel_pose=rel_pose,
                ref_images=ref_images,
                layout_canvas=layout_canvas
            )
            kwargs_with_uncond.pop("max_len", None)  # some do not take this.
            camera_param = kwargs_with_uncond.pop("camera_param")
            rel_pose = kwargs_with_uncond.pop("rel_pose")
            ref_images = kwargs_with_uncond.pop("ref_images")
            layout_canvas = kwargs_with_uncond.pop("layout_canvas")
            _images[0] = kwargs_with_uncond.pop("image")
            image = torch.cat(_images)
            bev_controlnet_kwargs = move_to(kwargs_with_uncond, self.device)

            if self.embedder is not None:
                image_tensor = rearrange(ref_images, "b n c h w -> (b n) c h w")
                # img: b c h w >> b l c
                ref_images = self.embedder(image_tensor)
                if self.image_proj_model:
                    ref_images = self.image_proj_model(ref_images)
                ref_images = rearrange(ref_images, "(b n) c l -> b n c l", n=N_cam)
            

        ###### BEV end ######

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # bs*2, 6, 4, 28, 50
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # controlnet(s) inference
                controlnet_t = t.unsqueeze(0)
                # guess_mode & classifier_free_guidance -> only guidance use controlnet
                # not guess_mode & classifier_free_guidance -> all use controlnet
                # guess_mode -> normal input, take effect in controlnet
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    controlnet_latent_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_latent_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                controlnet_t = controlnet_t.repeat(len(controlnet_latent_model_input))

                # fmt: off
                down_block_res_samples, mid_block_res_sample, \
                encoder_hidden_states_with_cam = self.controlnet(
                    controlnet_latent_model_input,
                    controlnet_t,
                    camera_param,  # for BEV
                    encoder_hidden_states=controlnet_prompt_embeds,
                    bev_hdmap=image,  # b, 3, 200, 200
                    rel_pose=rel_pose,
                    ref_images=ref_images,
                    layout_canvas=layout_canvas,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                    **bev_controlnet_kwargs, # for BEV
                )
                # fmt: on

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [
                        torch.cat([torch.zeros_like(d), d])
                        for d in down_block_res_samples
                    ]
                    mid_block_res_sample = torch.cat(
                        [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                    )
                    # add uncond encoder_hidden_states_with_cam here
                    encoder_hidden_states_with_cam = self.controlnet.add_uncond_to_emb(
                        prompt_embeds.chunk(2)[0], N_cam,
                        encoder_hidden_states_with_cam,
                    )

                # =============================================================
                # Strating from here, we use 4-dim data.
                # encoder_hidden_states_with_cam: (2b x N), 78, 768
                # latent_model_input: 2b, N, 4, 28, 50 -> 2b x N, 4, 28, 50
                latent_model_input = rearrange(
                    latent_model_input, 'b n ... -> (b n) ...')
                latents = rearrange(latents, 'b n ... -> (b n) ...')

                # predict the noise residual: 2bxN, 4, 28, 50
                additional_param = {}
                if hasattr(self.unet, "crossview_attn_type"):
                    additional_param['camera_params_raw'] = camera_params_raw
                noise_pred = self.unet(
                    latent_model_input,  # may with unconditional
                    t,
                    encoder_hidden_states=encoder_hidden_states_with_cam,
                    **additional_param,  # if use original unet, it cannot take kwargs
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    # for each: bxN, 4, 28, 50
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                # NOTE: is the scheduler use randomness, please handle the logic
                # for generator.
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # =============================================================
                # now we add dimension back, use 5-dim data.
                # NOTE: only `latents` is updated through the loop
                latents = rearrange(latents, '(b n) ... -> b n ...', n=N_cam)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        ###### BEV: here rebuild the shapes back. post-process still assume
        # latents, no need for b, n, 4, 28, 50
        # prompt_embeds, no need for b, len, 768
        # image, no need for b, c, 200, 200
        ##### BEV end

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

            # 10. Convert to PIL
            image = self.numpy_to_pil_double(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


    def prepare_latents_with_ref_img(self, image, timestep, batch_size, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            init_latents = image

        else:
            init_latents = self.vae.encode(image).latent_dist.sample()

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents