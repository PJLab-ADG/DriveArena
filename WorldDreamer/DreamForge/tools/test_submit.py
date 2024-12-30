import base64
import io
import copy
import os
from collections import defaultdict
import pickle
import random
import torch
import torchvision
from tqdm import tqdm
import shutil
import numpy as np

from PIL import ImageOps, Image
from moviepy.editor import *

from argparse import ArgumentParser
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path
import sys
from hydra import initialize, compose
from omegaconf import OmegaConf
from mmdet3d.datasets import build_dataset

sys.path.append(".")  # noqa
from dreamforge.runner.utils import concat_6_views, img_concat_h, img_concat_v
from dreamforge.misc.test_utils import (
    build_pipe, run_one_batch, run_one_batch_map, update_progress_bar_config, collate_fn_single, ListSetWrapper, partial
)

CAM_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
    ]
transparent_bg = False
target_map_size = 400
save_ori = False

def output_func(x): return concat_6_views(x, oneline=True)

def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)


class ImageNormalize: ## Important !!! check the mean and std which should be consistent with the dataset.
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.compose(img)


def load_pipe(resume_from_checkpoint, config_name="test_config_single", device='cuda'):
    try:
        initialize(version_base=None, config_path="../configs")
    except:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="../configs")
    output_dir = to_absolute_path(resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    overrides = original_overrides
    cfg = compose(config_name, overrides=overrides)
    cfg.resume_from_checkpoint = resume_from_checkpoint

    #### model ####
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    return pipe, cfg, weight_dtype


def load_data(cfg, pipe, val_dataset):
    #### datasets ####
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
    return val_dataloader


def custom_resize(img, out_size=(1600, 900)):
    if args.resize == 'bilinear':
        return img.resize(out_size, resample=Image.BICUBIC)
    else:
        return img


def generate_ref_with_single_pipe(val_input, weight_dtype):
    keys = ['meta_data', 'captions', "bev_map_with_aux", "pixel_values", "camera_param", 'kwargs', "bev_hdmap", "layout_canvas"]
    val_input_single = {}
    for key in keys:
        if key == 'meta_data':
            val_input_single[key] = {'metas': val_input[key]['metas'][:1], 'lidar2image': val_input[key]['lidar2image'][:1], 'gt_bboxes_3d': val_input[key]['gt_bboxes_3d'][:1]}
        elif key == 'kwargs':
            val_input_single[key] = {'bboxes_3d_data': {k:v[:1] for k,v in val_input[key]['bboxes_3d_data'].items()}}
        else:
            val_input_single[key] = val_input[key][:1]
    cfg_single.runner.validation_times = 1
    cfg_single.show_box = False
    gen_imgs_list = run_one_batch(cfg_single, pipe_single, val_input_single, weight_dtype)[3]

    # for i, x in enumerate(gen_imgs_list[0][0]):
    #     x.save(os.path.join(cfg.log_root, f'test_{i}.png'))
        
    # breakpoint()
    
    ref_image = torch.stack([ImageNormalize()(x) for x in gen_imgs_list[0][0]])
    ref_images = torch.stack([ref_image, ref_image.clone()])
    val_input['ref_values'] = ref_images
    
    return val_input


def generate_scene(val_dataloader, scene, overlaps, generator=None):
    #### start ####
    total_num = 0
    batch_index = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    os.makedirs(os.path.join(cfg.log_root, scene, 'frames'), exist_ok=True)
    all_ori_img_paths = []
    all_gen_img_paths = defaultdict(list)
    for val_input in val_dataloader:
        batch_index += 1
        batch_img_index = 0
        ori_img_paths = []
        gen_img_paths = defaultdict(list)

        ### change environment
        scene_type = scene.split('_')[-1]
        if scene_type == 'night':
            val_input['captions'] = ['A driving scene image at singapre-hollandvillage. night, clear, suburban, streetlights.']*len(val_input['captions'])
        elif scene_type == 'rainy':
            val_input['captions'] = ['A driving scene image at boston-seaport. rainy, cloudy, suburban, wet road.']*len(val_input['captions'])
        ###

        if batch_index > 1:
            val_input['ref_values'] = torch.stack(ref_images)
            if prev_pos is not None:
                val_input['can_bus'][0][:3] = prev_pos
                val_input['can_bus'][0][-1] = prev_angle 
            ref_can_bus[0][:3] = 0
            ref_can_bus[0][-1] = 0
            val_input['ref_can_bus'] = torch.stack(ref_can_bus)
            if is_overlap_condition and len(overlap_images):
                val_input['overlap_values'] = torch.stack(overlap_images)
        else:
            val_input = generate_ref_with_single_pipe(val_input, weight_dtype)
            
        return_tuples = run_one_batch_map(cfg, pipe, val_input, weight_dtype,
                                    transparent_bg=transparent_bg, generator=generator,
                                    map_size=target_map_size)
        generator, return_tuples = return_tuples[-1], return_tuples[:-1]

        if batch_index < len(val_dataloader.dataset):
            overlap_length = overlaps[batch_index]
            # ref_idxs = sorted(random.sample(range(cfg.dataset.data.val.candidate_length), cfg.model.ref_length))
            ref_idxs = [cfg.model.video_length-overlap_length-2, cfg.model.video_length-overlap_length-1]
            ref_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[3][idx][0]]) for idx in ref_idxs]
            ref_can_bus = [val_input['can_bus'][idx] for idx in ref_idxs]

            if is_overlap_condition:
                overlap_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[3][idx][0]]) for idx in range(cfg.model.video_length-overlap_length, cfg.model.video_length)]

            if overlap_length != 0:
                prev_pos, prev_angle = val_input['ref_can_bus'][0][:3], val_input['ref_can_bus'][0][-1]
                for idx in range(ref_idxs[-1]+1, cfg.model.video_length-overlap_length+1):
                    prev_pos += val_input['can_bus'][idx][:3]
                    prev_angle += val_input['can_bus'][idx][-1]
            else:
                prev_pos, prev_angle = None, None

        for _, ori_imgs, _, gen_imgs_list, _ in zip(*return_tuples):
            # # save ori
            if save_ori:
                if ori_imgs is not None:
                    ori_img = output_func(ori_imgs)
                    save_path = os.path.join(
                        cfg.log_root, scene, "frames",
                        f"{batch_index}_{batch_img_index}_ori_{total_num}.png")
                    ori_img.save(save_path)
                    ori_img_paths.append(save_path)

            # save gen
            gen_imgs = gen_imgs_list[0]
            for cam, img in zip(CAM_NAMES, gen_imgs):
                img = custom_resize(img)
                save_path = os.path.join(
                    cfg.log_root, scene, "frames",
                    f"{batch_index}_{batch_img_index}_gen_{total_num}_{cam}.png")
                img.save(save_path)
                gen_img_paths[cam].append(save_path)

            total_num += 1
            batch_img_index += 1

        if batch_index > 1:
            overlap_length = overlaps[batch_index-1]
            if save_ori:
                ori_img_paths = ori_img_paths[overlap_length:]
            for cam in CAM_NAMES:
                gen_img_paths[cam] = gen_img_paths[cam][overlap_length:]
        if save_ori:
            all_ori_img_paths.extend(ori_img_paths)
        for cam in CAM_NAMES:
            all_gen_img_paths[cam].extend(gen_img_paths[cam])

        # update bar
        progress_bar.update(cfg.runner.validation_times)

    if save_ori:
        make_video_with_filenames(
            all_ori_img_paths, os.path.join(
                cfg.log_root, scene, f"{scene.split('_')[0]}_ori.mp4"),
            fps=cfg.fps)
    for cam in CAM_NAMES:
        make_video_with_filenames(
            all_gen_img_paths[cam], os.path.join(
                cfg.log_root, scene, f"{scene.split('_')[0]}_{cam}.mp4"), fps=cfg.fps)

    shutil.rmtree(os.path.join(cfg.log_root, scene, 'frames'))

def main(mode='standarded'):
    video_length = cfg.model.video_length
    ref_length = cfg.model.ref_length
    candidate_length = cfg.model.ref_length

    val_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    )

    token_data_dict = val_dataset.token_data_dict
    if mode == 'standarded':
        with open(args.input, 'rb') as f:
            scene_tokens = pickle.load(f)['scene_tokens']

        start_idxs = [0, 5, 9]
        overlaps = [0, 2, 3]
        seeds = [42, 1024, 2345, 13123]
        prefixs = ['gen0', 'gen1', 'gen2', 'gen3']

        scene_clips = defaultdict(list)
        for scene in scene_tokens:
            for start in start_idxs:
                clip = [token_data_dict[token] for token in scene[start: start + video_length]]
                ref_idx = sorted(random.sample(range(max(start-candidate_length, 0), start), ref_length)) if start !=0 else [0]*ref_length
                ref = [token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip
                scene_clips[scene[0]].append(clip)

        for i, (k, v) in enumerate(scene_clips.items()):
            val_dataset.clip_infos = list(v)
            val_dataloader = load_data(cfg, pipe, val_dataset)
            for seed, prefix in zip(seeds, prefixs):
                cfg_single.seed = seed
                generate_scene(val_dataloader, k+'_'+prefix, overlaps=overlaps)

    elif mode == 'long':
        with open(args.input, 'rb') as f:
            scene_tokens = pickle.load(f)['scene_tokens']
        
        seeds = [1024, 13123, 42]
        prefixs = ['night', 'rainy', 'sunny']
        for i, scene in enumerate(scene_tokens):
            start_idxs = list(range(0, len(scene), 7))
            overlaps = [0]*len(start_idxs)
            if i < 2:
                start_idxs[-1] -= 3
                overlaps[-1] += 3
            else:
                start_idxs[-1] -= 4
                overlaps[-1] += 4
            
            scene_clips = []
            for start in start_idxs:
                clip = [token_data_dict[token] for token in scene[start: start + video_length]]
                ref_idx = sorted(random.sample(range(max(start-candidate_length, 0), start), ref_length)) if start !=0 else [0]*ref_length
                ref = [token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip
                scene_clips.append(clip)

            val_dataset.clip_infos = scene_clips
            val_dataloader = load_data(cfg, pipe, val_dataset)
            for seed, prefix in zip(seeds, prefixs):
                cfg_single.seed = seed
                generate_scene(val_dataloader, scene[0]+'_'+prefix, overlaps=overlaps)

    
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--resize", type=str, default='')
    parser.add_argument("-i", "--input", type=str, default='./data/workshop/nuscenes_interp_12Hz_infos_track2_eval_long.pkl')
    parser.add_argument("-o", "--output", type=str, default='./work_dirs/submit_scenarios/222x400')
    parser.add_argument("--model_single", type=str, default='./pretrained/dreamforge-s')
    parser.add_argument("--model", type=str, default='./pretrained/dreamforge-t')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()
    pipe_single, cfg_single, _ = load_pipe(args.model_single, config_name="test_config_single")
    pipe, cfg, weight_dtype = load_pipe(args.model, config_name="test_config")

    cfg.log_root = args.output
    is_overlap_condition = True
    cfg.fps = 12
    # cfg.runner.pipeline_param.guidance_scale=4
    main(mode='standarded' if 'long' not in os.path.basename(args.input) else 'long')



