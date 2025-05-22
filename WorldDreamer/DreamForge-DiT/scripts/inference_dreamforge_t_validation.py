import os
import gc
import sys
import time
import copy
from pprint import pformat
from datetime import timedelta
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        # just before torch_npu, let xformers know there is no gpu
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import dreamforgedit.utils.module_contrib

import colossalai
import torch.distributed as dist
from torch.utils.data import Subset
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from mmengine.runner import set_random_seed
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from mmcv.parallel import DataContainer

from dreamforgedit.acceleration.parallel_states import (
    set_sequence_parallel_group,
    get_sequence_parallel_group,
    set_data_parallel_group,
    get_data_parallel_group
)
from dreamforgedit.datasets import save_sample
from dreamforgedit.datasets.dataloader import prepare_dataloader
from dreamforgedit.datasets.dataloader import prepare_dataloader
from dreamforgedit.models.text_encoder.t5 import text_preprocessing
from dreamforgedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from dreamforgedit.utils.config_utils import parse_configs, define_experiment_workspace, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from dreamforgedit.utils.inference_utils import (
    apply_mask_strategy,
    get_save_path_name,
    concat_6_views_pt,
    add_null_condition,
    enable_offload,
)
from dreamforgedit.utils.misc import (
    reset_logger,
    is_distributed,
    is_main_process,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from dreamforgedit.utils.train_utils import sp_vae


TILING_PARAM = {
    "default": dict(),  # it is designed for CogVideoX's 720x480, 4.5 GB
    "384": dict(  # about 14.2 GB
        tile_sample_min_height = 384,  # should be 48n
        tile_sample_min_width = 720,  # should be 40n
    ),
}


class FakeCoordinator:
    def block_all(self):
        pass

    def is_master(self):
        return True
    
    def destroy(self):
        pass

    
def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    # == dataset config ==
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
        for (res, data_cfg_name), overrides in zip(
                cfg.data_cfg_names, cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)):
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)
    if cfg.get("use_train", False):
        cfg.dataset = dataset
        tag = cfg.get("tag", "")
        cfg.tag = "train" if tag == "" else f"{tag}_train"
    else:
        cfg.dataset = val_dataset
    # set img_collate_param
    if hasattr(cfg.dataset, "img_collate_param"):
        cfg.dataset.img_collate_param.is_train = False  # Important!
    else:
        for d in cfg.dataset.cfg:
            d[1].img_collate_param.is_train = False  # Important!
    cfg.batch_size = 1
    # for lower cpu memory in dataloading
    cfg.ignore_ori_imgs = cfg.get("ignore_ori_imgs", False)
    if cfg.ignore_ori_imgs:
        cfg.dataset.img_collate_param.drop_ori_imgs = True

    # for lower gpu memory in vae decoding
    cfg.vae_tiling = cfg.get("vae_tiling", None)

    # edit annotations
    if cfg.get("allow_class", None) != None:
        cfg.dataset.allow_class = cfg.allow_class
    if cfg.get("del_box_ratio", None) != None:
        cfg.dataset.del_box_ratio = cfg.del_box_ratio
    if cfg.get("drop_nearest_car", None) != None:
        cfg.dataset.drop_nearest_car = cfg.drop_nearest_car

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if USE_NPU:  # disable some kernels
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == init distributed env ==
    if is_distributed():
        # colossalai.launch_from_torch({})
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        coordinator = DistCoordinator()
        cfg.sp_size = dist.get_world_size()
        if cfg.sp_size > 1:
            DP_AXIS, SP_AXIS = 0, 1
            dp_size = dist.get_world_size() // cfg.sp_size
            pg_mesh = ProcessGroupMesh(dp_size, cfg.sp_size)
            dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
            sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
            set_sequence_parallel_group(sp_group)
            print(f"Using sp_size={cfg.sp_size}")
        else:
            # TODO: sequence_parallel_group unset!
            dp_group = dist.group.WORLD
        set_data_parallel_group(dp_group)
        enable_sequence_parallelism = cfg.sp_size > 1
    else:
        cfg.sp_size = 1
        coordinator = FakeCoordinator()
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init exp_dir ==
    cfg.outputs = cfg.get("outputs", "outputs/test")
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    cfg.save_dir = os.path.join(exp_dir, "generation")
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger ==
    logger = reset_logger(exp_dir)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    if cfg.get("val", None):
        validation_index = cfg.val.validation_index
        if validation_index == "all":
            raise NotImplementedError()
        cfg.num_sample = cfg.val.get("num_sample", 1)
        cfg.scheduler = cfg.val.get("scheduler", cfg.scheduler)
    else:
        validation_index = cfg.get("validation_index", "all")

    # == build dataset ==
    logger.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    if validation_index == "even":
        idxs = list(range(0, len(dataset), 2))
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif validation_index == "odd":
        idxs = list(reversed(list(range(1, len(dataset), 2))))  # reversed!
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif 'scene' in validation_index:
        dataset.clip_infos = dataset.scene_clips[int(validation_index.split('_')[1])]
    elif validation_index == "all":
        dataset.clip_infos = [dataset.valid_clips[i][0] for i in range(len(dataset.valid_clips))]
    elif validation_index != "all":
        dataset = torch.utils.data.Subset(dataset, validation_index)
    logger.info(f"Your validation index: {validation_index}")
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,  # changed
        drop_last=False,  # changed
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    # NOTE: set to true/false,
    # https://github.com/huggingface/transformers/issues/5486
    # if the program gets stuck, try set it to false
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    if cfg.vae_tiling:
        vae.module.enable_tiling(**TILING_PARAM[str(cfg.vae_tiling)])
        logger.info(f"VAE Tiling is enabled with {TILING_PARAM[str(cfg.vae_tiling)]}")

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    model_single = (
        build_module(
            cfg.model_single,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    cfg.cpu_offload = cfg.get("cpu_offload", False)
    if cfg.cpu_offload:
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)
    # == load prompts ==
    # prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)

    # == prepare arguments ==
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    assert num_sample == 1

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == Iter over all samples ==
    start_step = 0
    assert batch_size == 1
    sampler.set_epoch(0)
    dataloader_iter = iter(dataloader)
    with tqdm(
        enumerate(dataloader_iter, start=start_step),
        desc=f"Generating",
        disable=not coordinator.is_master() or not verbose,
        initial=start_step,
        total=num_steps_per_epoch,
    ) as pbar:
        for i, batch in pbar:
            if cfg.ignore_ori_imgs:
                B, T, NC = batch["pixel_values_shape"].tolist()[:3]
                latent_size = vae.get_latent_size(
                    (T, *batch["pixel_values_shape"].tolist()[-2:]))
            else:
                B, T, NC = batch["pixel_values"].shape[:3]
                latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))
                # == prepare batch prompts ==
                x = batch.pop("pixel_values").to(device, dtype)
                x = rearrange(x, "B T NC C ... -> (B NC) C T ...")  # BxNC, C, T, H, W
            y = batch.pop("captions")[0]  # B, just take first frame
            maps = batch.pop("bev_hdmap").to(device, dtype)  # B, T, C, H, W
            bbox = batch.pop("bboxes_3d_data")
            layouts = batch.pop("layout_canvas")

            # B, T, NC, 3, 7
            cams = batch.pop("camera_param").to(device, dtype)
            cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
            rel_pos = batch.pop("frame_emb").to(device, dtype)
            rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4

            # variable for inference
            batch_prompts = y

            # == model input format ==
            model_args = {}
            model_args["maps"] = maps
            model_args["layouts"] = layouts
            model_args["bbox"] = bbox
            model_args["cams"] = cams
            model_args["rel_pos"] = rel_pos
            model_args["fps"] = batch.pop('fps')
            model_args['drop_cond_mask'] = torch.ones((B))  # camera
            model_args['drop_frame_mask'] = torch.ones((B, T))  # box & rel_pos
            model_args["height"] = batch.pop("height")
            model_args["width"] = batch.pop("width")
            model_args["num_frames"] = batch.pop("num_frames")
            model_args = move_to(model_args, device=device, dtype=dtype)
            # no need to move these
            model_args["mv_order_map"] = cfg.get("mv_order_map")
            model_args["t_order_map"] = cfg.get("t_order_map")
            model_args["img_metas"] = batch.pop("meta_data")

            # == Iter over number of sampling for one prompt ==
            save_fps = int(model_args['fps'][0])

            scene_token = model_args['img_metas']['metas'][0][0].data['token']
            
            gc.collect()
            torch.cuda.empty_cache()
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=y[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=0,
                )
                for idx in range(len(y))
            ]
            if cfg.get("force_daytime", False):
                batch_prompts[0] = batch_prompts[0].lower()
                batch_prompts[0] = "Daytime. " + batch_prompts[0]
                # exclude rain
                batch_prompts[0] = batch_prompts[0].replace("rain", "sunny")
                batch_prompts[0] = batch_prompts[0].replace("water reflections", "")
                batch_prompts[0] = batch_prompts[0].replace("reflections in water", "")
                batch_prompts[0] = batch_prompts[0].replace(" with umbrellas", "")
                batch_prompts[0] = batch_prompts[0].replace(" with umbrella", "")
                batch_prompts[0] = batch_prompts[0].replace(" holds umbrella", "")
                # exclude night
                batch_prompts[0] = batch_prompts[0].replace("night", "")
                batch_prompts[0] = batch_prompts[0].replace(" in dark", "")
                batch_prompts[0] = batch_prompts[0].replace(" dark", "")
                batch_prompts[0] = batch_prompts[0].replace(" difficult lighting", "")
                # city
                batch_prompts[0] = batch_prompts[0].replace("boston-seaport", "singapore-onenorth")
                batch_prompts[0] = batch_prompts[0].replace("singapore-hollandvillage", "singapore-onenorth")
                neg_prompts = ["Rain, Night, water reflections, umbrella"]
            elif cfg.get("force_rainy", False):
                if "rain" not in batch_prompts[0].lower():
                    batch_prompts[0] = "A driving scene image at boston-seaport. Rain. water reflections."
                neg_prompts = ["Daytime. night, onenorth, queenstown"]
            elif cfg.get("force_night", False):
                if "night" not in batch_prompts[0].lower():
                    batch_prompts[0] = "A driving scene image at singapore-hollandvillage. Night, congestion. difficult lighting. very dark."
                neg_prompts = ["Daytime. rain, boston-seaport"]
            else:
                neg_prompts = None

            video_clips = []
            # == sampling ==
            torch.manual_seed(1024)  # NOTE: not sure how to handle loop, just change here.
            z = torch.randn(len(batch_prompts), vae.out_channels * NC, *latent_size, device=device, dtype=dtype)

            # == sample box ==
            if bbox is not None:
                # null set values to all zeros, this should be safe
                bbox = add_box_latent(bbox, B, NC, T, model.sample_box_latent)
                # overwrite!
                new_bbox = {}
                for k, v in bbox.items():
                    new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
                model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)

            # == add null condition ==
            # y is handled by scheduler.sample
            if cfg.scheduler.type == "dpm-solver" and cfg.scheduler.cfg_scale == 1.0 or (
                cfg.scheduler.type in ["rflow-slice",]
            ):
                _model_args = copy.deepcopy(model_args)
            else:
                _model_args = add_null_condition(
                    copy.deepcopy(model_args),
                    model.camera_embedder.uncond_cam.to(device),
                    model.frame_embedder.uncond_cam.to(device),
                    prepend=(cfg.scheduler.type == "dpm-solver"),
                )

            # == inference ==
            if i < 150:  # 只针对第一个样本进行特殊处理
                # 第一阶段：使用model_single生成第一帧
                masks_single = torch.full((1, z.shape[2]), True, dtype=torch.bool, device=device)
                masks_single[0, :1] = True 
                
                # 获取原始图像的第一帧作为条件
                x_encoded = rearrange(vae.encode(x[:, :, :1]), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                z_single = z.clone()
                z_single = z_single[:, :, :1]
                
                # 为model_single准备只包含第一帧的参数
                _model_args_single = copy.deepcopy(_model_args)
                
                # 修改_model_args_single中的时序维度，仅保留第一帧
                _model_args_single["maps"] = _model_args_single["maps"][:, :1]
                
                _model_args_single["layouts"] = _model_args_single["layouts"][:, :1]
                
                _model_args_single["cams"] = _model_args_single["cams"][:, :1]
                
                _model_args_single["rel_pos"] = _model_args_single["rel_pos"][:, :1]

                _model_args_single["num_frames"] = torch.tensor([1.], device=device, dtype=dtype)
                
                _model_args_single["drop_frame_mask"] = _model_args_single["drop_frame_mask"][:, :1]
                
                # 处理bbox（可能包含多个键，每个键都有时序维度）
                if "bbox" in _model_args_single and _model_args_single["bbox"] is not None:
                    for k in _model_args_single["bbox"]:
                        _model_args_single["bbox"][k] = _model_args_single["bbox"][k][:, :1]

                # 在处理img_metas时，需要保持DataContainer结构
                for k in _model_args_single["img_metas"]:
                    if k in ['lidar2image', 'img2lidars']:
                        _model_args_single["img_metas"][k] = _model_args_single["img_metas"][k][0][:1]  # 只保留第一帧
                    else:
                        _model_args_single["img_metas"][k] = _model_args_single["img_metas"][k][0][:1]  # 只保留第一帧
                masks_single = masks_single[:, :1]

                # 使用model_single和修改后的参数生成第一帧
                samples_single = scheduler.sample(
                    model_single,
                    text_encoder,
                    z=z_single,
                    prompts=batch_prompts,
                    neg_prompts=neg_prompts,
                    device=device,
                    additional_args=_model_args_single,
                    progress=verbose >= 1,
                    mask=masks_single,
                )
                
                # 将结果解码为图像
                samples_single = rearrange(samples_single, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
                if cfg.sp_size > 1:
                    samples_single = sp_vae(
                        samples_single.to(dtype),
                        partial(vae.decode, num_frames=_model_args_single["num_frames"]),
                        get_sequence_parallel_group(),
                    )
                else:
                    samples_single = vae.decode(samples_single.to(dtype), num_frames=_model_args_single["num_frames"])
                
                # 提取第一帧作为参考样本
                first_frame = samples_single[:, :, :1].clone()
                
                # 释放model_single的内存

                
                # 第二阶段：使用model生成所有帧，使用生成的第一帧作为条件
                masks = torch.full((1, z.shape[2]), True, dtype=torch.bool, device=device)
                masks[0, :1] = False  # 只使用第一帧作为条件
                
                # 对生成的第一帧进行编码
                first_frame_encoded = rearrange(vae.encode(first_frame), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                z[0, :, :1] = first_frame_encoded
                
                # 使用model生成完整视频
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    neg_prompts=neg_prompts,
                    device=device,
                    additional_args=_model_args,
                    progress=verbose >= 1,
                    mask=masks,
                )
            else:
                # 其他样本正常处理
                del model_single, samples_single
                gc.collect()
                torch.cuda.empty_cache()
                masks = torch.full((1, z.shape[2]), True, dtype=torch.bool, device=device)
                if i > 0 and T > 1:
                    masks[0, :3] = False
                    x_encoded = rearrange(vae.encode(ref_samples[:, :, -9:]), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                    z[0, :, :3] = x_encoded
                else:
                    masks[0, :1] = False
                    x_encoded = rearrange(vae.encode(x[:, :, :1]), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
                    z[0, :, :1] = x_encoded
                
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    neg_prompts=neg_prompts,
                    device=device,
                    additional_args=_model_args,
                    progress=verbose >= 1,
                    mask=masks,
                )

            # 解码为图像
            samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
            if cfg.sp_size > 1:
                samples = sp_vae(
                    samples.to(dtype),
                    partial(vae.decode, num_frames=_model_args["num_frames"]),
                    get_sequence_parallel_group(),
                )
            else:
                samples = vae.decode(samples.to(dtype), num_frames=_model_args["num_frames"])
            ref_samples = samples.clone()

            samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
            if cfg.cpu_offload:
                last_hook.offload()
            if is_main_process():
                vid_samples = []
                for sample in samples:
                    vid_samples.append(
                        concat_6_views_pt(sample, oneline=False)
                    )
                samples = torch.stack(vid_samples, dim=0)
                video_clips.append(samples)
                del vid_samples
            del samples
            coordinator.block_all()

            # == save samples ==
            torch.cuda.empty_cache()
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 1:
                        logger.info(f"Prompt: {batch_prompt}")
                        if neg_prompts is not None:
                            logger.info(f"Neg-prompt: {neg_prompts[idx]}")
                    save_path = save_paths[idx]
                    video = [video_clips[0][idx]]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        high_quality=True,
                        verbose=verbose >= 2,
                        save_per_n_frame=cfg.get("save_per_n_frame", -1),
                        force_image=cfg.get("force_image", False),
                    )
                    view_names = [
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT",
                        "CAM_FRONT_RIGHT", 
                        "CAM_BACK_RIGHT",
                        "CAM_BACK",
                        "CAM_BACK_LEFT"
                    ]
                    

                    # 获取单个样本 - [3, 17, 448, 800]
                    sample = ref_samples
                    
                    # 获取分辨率
                    resolution = f"{int(sample.shape[3])}x{int(sample.shape[4])}"
                    save_dir_scene = os.path.join(save_dir, resolution, scene_token)
                    os.makedirs(save_dir_scene, exist_ok=True)
                    
                    # 对于形状 [6, 3, 17, 448, 800] 的ref_samples
                    # 其中6是视角，3是通道，17是帧数，448x800是分辨率
                    for view_idx, view_name in enumerate(view_names):
                        if view_idx < ref_samples.shape[0]:  # 确保视角索引有效
                            # 提取单个视角的所有帧
                            view_sample = ref_samples[view_idx]  # [3, 17, 448, 800]
                            
                            # 限制帧数（如果需要）
                            max_frames = min(16, view_sample.shape[1])
                            view_sample = view_sample[:, :max_frames]  # [3, max_frames, 448, 800]
                            
                            # 创建视频文件名
                            video_path = os.path.join(save_dir_scene, f"{scene_token}_{view_name}.mp4")
                            
                            # 保存视频
                            try:
                                save_sample(
                                    view_sample,  # 已经是正确的 [C, T, H, W] 形状
                                    fps=save_fps,
                                    save_path=video_path,
                                    high_quality=True,
                                    verbose=verbose >= 2,
                                    save_per_n_frame=-1,
                                    with_postfix=False,
                                    force_image=False,
                                )
                                if verbose >= 1:
                                    logger.info(f"已保存视角 {view_name} 到 {video_path}")
                            except Exception as e:
                                logger.error(f"保存视角 {view_name} 失败: {e}, 形状: {view_sample.shape}")

            del video_clips
            coordinator.block_all()
            
            # save_gt
            if is_main_process() and not cfg.ignore_ori_imgs:
                torch.cuda.empty_cache()
                samples = rearrange(x, "(B NC) C T H W -> B NC C T H W", NC=NC)
                for idx, sample in enumerate(samples):
                    vid_sample = concat_6_views_pt(sample, oneline=False)
                    save_path = save_sample(
                        vid_sample,
                        fps=save_fps,
                        save_path=os.path.join(save_dir, f"gt_{start_idx + idx:04d}"),
                        high_quality=True,
                        verbose=verbose >= 2,
                        save_per_n_frame=cfg.get("save_per_n_frame", -1),
                        force_image=cfg.get("force_image", False),
                    )
                del samples, vid_sample
            coordinator.block_all()
            start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx - cfg.get("start_index", 0), save_dir)
    coordinator.destroy()


if __name__ == "__main__":
    main()


