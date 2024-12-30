from typing import Tuple, Union, List
import os
import logging
from hydra.core.hydra_config import HydraConfig
from functools import partial
from omegaconf import OmegaConf
from omegaconf import DictConfig
from PIL import Image

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from einops import rearrange

from mmdet3d.datasets import build_dataset
from diffusers import UniPCMultistepScheduler
import accelerate
from accelerate.utils import set_seed

from dreamforge.dataset import collate_fn, ListSetWrapper, collate_fn_single
from dreamforge.pipeline.pipeline_bev_controlnet import (
    StableDiffusionBEVControlNetPipeline,
    BEVStableDiffusionPipelineOutput,
)
from dreamforge.runner.utils import (
    visualize_map, img_m11_to_01, show_box_on_views
)
from dreamforge.misc.common import load_module, move_to



def insert_pipeline_item(cfg: DictConfig, search_type, item=None) -> None:
    if item is None:
        return
    assert OmegaConf.is_list(cfg)
    ori_cfg: List = OmegaConf.to_container(cfg)
    for index, _it in enumerate(cfg):
        if _it['type'] == search_type:
            break
    else:
        raise RuntimeError(f"cannot find type: {search_type}")
    ori_cfg.insert(index + 1, item)
    cfg.clear()
    cfg.merge_with(ori_cfg)


def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe, "_progress_bar_config"):
        config = pipe._progress_bar_config
        config.update(kwargs)
    else:
        config = kwargs
    pipe.set_progress_bar_config(**config)


def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)


def build_pipe(cfg, device):
    weight_dtype = torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    pipe_param = {}

    model_cls = load_module(cfg.model.model_module)
    controlnet_path = os.path.join(
        cfg.resume_from_checkpoint, cfg.model.controlnet_dir)
    if not os.path.exists(controlnet_path):
        controlnet_path = os.path.join(
            cfg.model.pretrained_dreamforge, cfg.model.controlnet_dir)
    logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
    controlnet = model_cls.from_pretrained(
        controlnet_path, torch_dtype=weight_dtype)
    controlnet.eval()  # from_pretrained will set to eval mode by default
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=weight_dtype)
        if hasattr(cfg.model, 'sc_attn_index'):
            logging.warn(f"We reset sc_attn_index from config.")
            for mod in unet.modules():
                if hasattr(mod, "_sc_attn_index"):
                    mod._sc_attn_index = OmegaConf.to_container(
                        cfg.model.sc_attn_index, resolve=True)
        unet.eval()
        pipe_param["unet"] = unet

    if hasattr(cfg.model, "can_bus_embedder"):
        can_bus_embedder_cls = load_module(cfg.model.can_bus_embedder_cls)
        can_bus_embedder_param = OmegaConf.to_container(cfg.model.can_bus_embedder, resolve=True)
        can_bus_embedder = can_bus_embedder_cls(**can_bus_embedder_param)
        can_bus_embedder_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.can_bus_embedder_dir)
        state_dict = torch.load(os.path.join(can_bus_embedder_path, "can_bus_embedder_model.bin"), map_location='cpu')
        can_bus_embedder.load_state_dict(state_dict)
        can_bus_embedder.type(torch.cuda.HalfTensor)
        can_bus_embedder.eval()
        pipe_param["can_bus_embedder"] = can_bus_embedder
        logging.info(f"Loading can_bus_embedder from {can_bus_embedder_path} with {can_bus_embedder_cls}")

    if hasattr(cfg.model, "scene_embedder_cls"):
        scene_embedder_cls = load_module(cfg.model.scene_embedder_cls)
        scene_embedder_param = OmegaConf.to_container(cfg.model.scene_embedder, resolve=True)
        scene_embedder = scene_embedder_cls(**scene_embedder_param)
        scene_embedder_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.scene_embedder_dir)
        if not os.path.exists(scene_embedder_path):
            scene_embedder_path = os.path.join(
                cfg.model.pretrained_dreamforge, cfg.model.scene_embedder_dir)
        state_dict = torch.load(os.path.join(scene_embedder_path, "scene_embedder_model.bin"), map_location='cpu')
        scene_embedder.load_state_dict(state_dict)
        scene_embedder.type(torch.cuda.FloatTensor)
        scene_embedder.eval()
        pipe_param["scene_embedder"] = scene_embedder
        logging.info(f"Loading scene_embedder from {scene_embedder_path} with {scene_embedder_cls}")

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype


def prepare_all(cfg, device='cuda', need_loader=True):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####
    val_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    )

    #### dataloader ####
    if cfg.runner.validation_index != "all":
        val_dataset = ListSetWrapper(val_dataset, cfg.runner.validation_index)

    if hasattr(cfg.model, "ref_length"):
        assert cfg.runner.validation_batch_size == 1, "Do not support more."
    collate_fn_param = {
        "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        "bbox_add_num": cfg.runner.bbox_add_num,
    }

    if hasattr(cfg.model, "ref_length"):
        collate_fn_param['ref_length'] = cfg.model.ref_length

    def _collate_fn(examples, *args, **kwargs):
        if hasattr(cfg.model, "ref_length"):
            return collate_fn_single(examples[0], *args, **kwargs)
        else:
            return collate_fn_single(examples, *args, **kwargs)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(_collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype


def new_local_seed(global_generator):
    local_seed = torch.randint(
        0x7ffffffffffffff0, [1], generator=global_generator).item()
    logging.debug(f"Using seed: {local_seed}")
    return local_seed


def run_one_batch_pipe(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    pixel_values: torch.FloatTensor,  # useless
    captions: Union[str, List[str]],
    bev_map_with_aux: Union[torch.FloatTensor, List[torch.FloatTensor]],
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    meta_data,
    conditional_values=None,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    device = pipe.device
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.Generator(
                        device=device).manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.Generator(
                    device=device).manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed)
                    for _ in range(batch_size)]
            else:
                generator = torch.Generator(device=device).manual_seed(cfg.seed)

    if conditional_values is not None:
        conditional_length, N_cam = conditional_values.shape[:2]
        with torch.no_grad():
            latents = pipe.vae.encode(
                rearrange(conditional_values, "b n c h w -> (b n) c h w").to(
                    dtype=pipe.vae.dtype, device=pipe._execution_device
                )
            ).latent_dist.mean
            latents = latents * pipe.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

    pipeline_param = {k: v for k, v in cfg.runner.pipeline_param.items()}
    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        if conditional_values is not None:
            conditional_latents = [[None,] * N_cam for _ in range(batch_size)]
            for b in range(conditional_length):
                for j in range(N_cam):
                    conditional_latents[b][j] = latents[b, j]
        else:
            conditional_latents = None
            
        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            image=bev_map_with_aux,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            conditional_latents=conditional_latents,
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            img_metas=meta_data,
            **pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list


def run_one_batch(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions']}")

    # map
    map_imgs = []
    if "bev_map_with_aux" in val_input:
        for bev_map in val_input["bev_map_with_aux"]:
            map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
            map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)

    if 'with_layout_canvas' in cfg.model.controlnet and cfg.model.controlnet.with_layout_canvas:
        bev_hdmap = val_input["bev_hdmap"].to(weight_dtype)
        layout_canvas = val_input["layout_canvas"].to(weight_dtype)
        controlnet_image = [bev_hdmap, layout_canvas]
    else:
        controlnet_image = val_input["bev_map_with_aux"].to(weight_dtype)

    conditional_values = val_input['conditional_values'].to(weight_dtype) if 'conditional_values' in val_input else None

    # 3-dim list: B, Times, views
    gen_imgs_list = run_one_batch_pipe_func(
        cfg, pipe, val_input['pixel_values'], val_input['captions'],
        controlnet_image, camera_param, val_input['kwargs'], val_input['meta_data'], conditional_values=conditional_values, 
        global_generator=global_generator)

    # save gen with box
    gen_imgs_wb_list = [None for _ in gen_imgs_list]
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list[bi] = [
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ]

    return map_imgs, ori_imgs, ori_imgs_with_box, gen_imgs_list, gen_imgs_wb_list


def run_one_batch_pipe_map(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    ref_values: torch.FloatTensor,
    captions: Union[str, List[str]],
    bev_hdmap: torch.FloatTensor, 
    layout_canvas: torch.FloatTensor,
    can_bus: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    meta_data=None,
    overlap_values=None,
    global_generator=None,
    generator=None,
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam
        conditional_latents: List[List[torch.FloatTensor]]

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    device = pipe.device
    if generator is None and cfg.seed is not None:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.Generator(
                        device=device).manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.Generator(
                    device=device).manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed)
                    for _ in range(batch_size)]
            elif 'overlap_same_noise' in cfg and cfg.overlap_same_noise:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed+i)
                    for i in range(batch_size)]
            else:
                generator = torch.Generator(device=device).manual_seed(cfg.seed)

    if overlap_values is not None:
        overlap_length, N_cam = overlap_values.shape[:2]
        with torch.no_grad():
            latents = pipe.vae.encode(
                rearrange(overlap_values, "b n c h w -> (b n) c h w").to(
                    dtype=pipe.vae.dtype, device=pipe._execution_device
                )
            ).latent_dist.mean
            latents = latents * pipe.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

    pipeline_param = {k: v for k, v in cfg.runner.pipeline_param.items()}
    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        if overlap_values is not None:
            conditional_latents = [[None,] * N_cam for _ in range(batch_size)]
            for b in range(overlap_length):
                for j in range(N_cam):
                    conditional_latents[b][j] = latents[b, j]
        else:
            conditional_latents = None

        if cfg.runner.pipeline_param.init_noise == "both":
            if pipeline_param['init_noise'] == "both":
                pipeline_param['init_noise'] = "same"
            elif pipeline_param['init_noise'] == "same":
                pipeline_param['init_noise'] = "rand"
            elif pipeline_param['init_noise'] == "rand":
                pipeline_param['init_noise'] = "same"
        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            image=[bev_hdmap, layout_canvas],
            ref_image=ref_values,
            can_bus=can_bus,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            conditional_latents=conditional_latents,
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            img_metas=meta_data,
            **pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list, generator


def run_one_batch_map(cfg, pipe, val_input, weight_dtype, generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe_map,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions']}")

    # map
    map_imgs = []
    if "bev_map_with_aux" in val_input:
        for bev_map in val_input["bev_map_with_aux"]:
            map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
            map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)
    ref_image = val_input["ref_values"].to(weight_dtype)
    ref_can_bus = val_input["ref_can_bus"].to(weight_dtype)
    can_bus =  val_input["can_bus"].to(weight_dtype)
    can_bus = torch.cat([ref_can_bus, can_bus], dim=0)
    bev_hdmap = val_input['bev_hdmap'].to(weight_dtype)
    layout_canvas = val_input['layout_canvas'].to(weight_dtype)
    overlap_values = val_input['overlap_values'].to(weight_dtype) if 'overlap_values' in val_input else None
    # 3-dim list: B, Times, views
    gen_imgs_list, generator = run_one_batch_pipe_func(
        cfg, pipe, ref_image, val_input['captions'],
        bev_hdmap, layout_canvas, can_bus, camera_param, val_input['kwargs'], meta_data=val_input['meta_data'], overlap_values=overlap_values, generator=generator)

    # save gen with box
    gen_imgs_wb_list = [None for _ in gen_imgs_list]
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list[bi] = [
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ]

    return map_imgs, ori_imgs, ori_imgs_with_box, gen_imgs_list, gen_imgs_wb_list, generator