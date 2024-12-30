from typing import Tuple, List
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate.tracking import GeneralTracker
from diffusers import UniPCMultistepScheduler

from dreamforge.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
    img_concat_v,
)
from dreamforge.runner.base_validator import (
    BaseValidator,
)
from dreamforge.misc.common import move_to
from dreamforge.misc.test_utils import draw_box_on_imgs
from dreamforge.pipeline.pipeline_bev_controlnet import (
    BEVStableDiffusionPipelineOutput,
)
from dreamforge.dataset.utils import collate_fn_single
from dreamforge.networks.unet_addon_rawbox import BEVControlNetModel


def format_image(image_list):
    formatted_images = []
    for image in image_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(
        to_pil_image(formatted_images))
    return formatted_images


class BaseTValidator(BaseValidator):
    def construct_visual(self, images_list, val_input, with_box):
        frame_list = []
        frame_list_wb = []
        for idx, framei in enumerate(images_list):
            frame = concat_6_views(framei, oneline=True)
            if with_box:
                frame_with_box = concat_6_views(
                    draw_box_on_imgs(
                        self.cfg, idx, val_input, framei),
                    oneline=True)
            frame_list.append(frame)
            frame_list_wb.append(frame_with_box)
        frames = img_concat_v(*frame_list)
        if with_box:
            frames_wb = img_concat_v(*frame_list_wb)
        else:
            frames_wb = None
        return frames, frames_wb

    def prepare_pipe(self, controlnet, unet, scene_embedder, can_bus_embedder, weight_dtype, device):
        controlnet.eval()  # important !!!
        unet.eval()
        can_bus_embedder.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            controlnet=controlnet,
            scene_embedder=scene_embedder,
            can_bus_embedder=can_bus_embedder,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    def validate(
        self,
        controlnet: BEVControlNetModel,
        unet,
        scene_embedder,
        can_bus_embedder,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info(f"[{self.__class__.__name__}] Running validation... ")
        torch.cuda.empty_cache()
        pipeline = self.prepare_pipe(controlnet, unet, scene_embedder, can_bus_embedder, weight_dtype, device)

        image_logs = []
        total_run_times = len(
            self.cfg.runner.validation_index) * self.cfg.runner.validation_times
        if self.cfg.runner.pipeline_param['init_noise'] == 'both':
            total_run_times *= 2
        progress_bar = tqdm(
            range(0, total_run_times), desc="Val Steps")

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = collate_fn_single(
                raw_data, self.cfg.dataset.template, is_train=False,
                bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
                ref_length=self.cfg.model.ref_length,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])
            camera_param = val_input["camera_param"].to(weight_dtype)
            bev_hdmap = val_input["bev_hdmap"].to(weight_dtype)
            layout_canvas = val_input['layout_canvas'].to(weight_dtype)

            ref_image = val_input["ref_values"].to(weight_dtype)
            ref_can_bus = val_input["ref_can_bus"].to(weight_dtype)
            can_bus = val_input["can_bus"].to(weight_dtype)
            can_bus = torch.cat([ref_can_bus, can_bus], dim=0)

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            def run_once(pipe_param):
                for _ in range(self.cfg.runner.validation_times):
                    with torch.autocast("cuda"):
                        image: BEVStableDiffusionPipelineOutput = pipeline(
                            prompt=val_input["captions"],
                            image=[bev_hdmap, layout_canvas],
                            ref_image=ref_image,
                            can_bus=can_bus,
                            camera_param=camera_param,
                            height=self.cfg.dataset.image_size[0],
                            width=self.cfg.dataset.image_size[1],
                            generator=generator,
                            bev_controlnet_kwargs=val_input["kwargs"],
                            img_metas=val_input['meta_data'],
                            **pipe_param,
                        )
                    gen_frames, gen_frames_wb = self.construct_visual(
                        image.images, val_input,
                        self.cfg.runner.validation_show_box)
                    gen_list.append(gen_frames)
                    if self.cfg.runner.validation_show_box:
                        gen_wb_list.append(gen_frames_wb)

                    progress_bar.update(1)

            # for each input param, we generate several times to check variance.
            gen_list, gen_wb_list = [], []
            pipeline_param = {
                k: v for k, v in self.cfg.runner.pipeline_param.items()}
            if self.cfg.runner.pipeline_param['init_noise'] != 'both':
                run_once(pipeline_param)
            else:
                pipeline_param['init_noise'] = "same"
                run_once(pipeline_param)
                pipeline_param['init_noise'] = "rand"
                run_once(pipeline_param)

            # make image for 6 views and save to dict
            ori_imgs = [[
                to_pil_image(img_m11_to_01(val_input["pixel_values"][j][i]))
                for i in range(6)
            ] for j in range(self.cfg.model.video_length)]
            ori_img, ori_img_wb = self.construct_visual(
                ori_imgs, val_input, True)
            map_img_np = visualize_map(
                self.cfg, val_input["bev_map_with_aux"][0])
            image_logs.append(
                {
                    "map_img_np": map_img_np,  # condition
                    "gen_img_list": gen_list,  # output
                    "gen_img_wb_list": gen_wb_list,  # output
                    "ori_img": ori_img,  # input
                    "ori_img_wb": ori_img_wb,  # input
                    "validation_prompt": val_input["captions"][0],
                }
            )
        self._save_image(step, image_logs, trackers)

    def _save_image(self, step, image_logs, trackers):
        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    map_img_np = log["map_img_np"]
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_image([log["ori_img"]])
                    tracker.writer.add_image(
                        "[ori]" + validation_prompt, formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_image(log["gen_img_list"])
                    tracker.writer.add_image(
                        "[gen]" + validation_prompt, formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_image([log["ori_img_wb"]])
                    tracker.writer.add_image(
                        "[ori]" + validation_prompt + "(with box)",
                        formatted_images, step, dataformats="HWC")

                    formatted_images = format_image(log["gen_img_wb_list"])
                    tracker.writer.add_image(
                        "[gen]" + validation_prompt + "(with box)",
                        formatted_images, step, dataformats="HWC")

                    tracker.writer.add_image(
                        "map: " + validation_prompt, map_img_np, step,
                        dataformats="HWC")
            else:
                logging.warn(
                    f"image logging not implemented for {tracker.name}")
        return image_logs
