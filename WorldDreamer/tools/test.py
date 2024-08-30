import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from moviepy.editor import *

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from projects.dreamer.runner.utils import concat_6_views, img_concat_h, img_concat_v
from projects.dreamer.utils.test_utils import prepare_all, run_one_batch
from data.demo_data.img_style import style_dict

target_map_size = 400

def output_func(x): return concat_6_views(x, oneline=True)

def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)

transform1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                 ])


@hydra.main(version_base=None, config_path="../configs", config_name="test_config")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "../hydra/overrides.yaml")
    )
    current_overrides = HydraConfig.get().overrides.task

    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)

    logging.info(f"Your validation index: {cfg.runner.validation_index}")

    #### setup everything ####
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    batch_index = -1
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    os.makedirs(os.path.join(cfg.log_root, "frames"), exist_ok=True)
    gen_ref = None
    for val_input in val_dataloader:
        batch_index += 1
        batch_img_index = 0
        if cfg.runner.validation_index in ['demo', 'all']:
            curr_index = batch_index
        else:
            curr_index = cfg.runner.validation_index[batch_index]

        ori_img_paths = []
        gen_img_paths = {}
        if cfg.runner.validation_index == 'demo' and batch_index == 0:
            val_input["ref_images"][0, ...] = style_dict('boston', cfg.dataset.dataset_root_nuscenes)
        elif val_input['meta_data']['metas'][0].data.get('is_first_frame', False):
            print(curr_index)
            pass
        elif gen_ref is None:
            pass
        else:
            val_input["ref_images"][0, ...] = gen_ref

        # You can change the description by the folowing code.
        # val_input['captions'] = ['A driving scene image at boston-seaport. daytime, rainy, downtown, straight road, white buildings, construction zone.']
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=cfg.transparent_bg,
                                      map_size=target_map_size)

        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save map
            map_img.save(
                os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_map.png",
                )
            )

            # save ori
            if ori_imgs is not None:
                ori_img = output_func(ori_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_ori.png",
                )
                ori_img.save(save_path)
                ori_img_paths.append(save_path)

            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = output_func(gen_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_gen{ti}.png",
                )
                gen_img.save(save_path)
                if ti in gen_img_paths:
                    gen_img_paths[ti].append(save_path)
                else:
                    gen_img_paths[ti] = [save_path]
            
            # process ref img
            gen_imgs = gen_imgs_list[0]
            ref_image_list = []
            for cam_i in range(6):
                img_i = gen_imgs[cam_i]
                img_i = np.array(img_i)
                img_i = transform1(img_i)
                ref_image_list.append(img_i)
            gen_ref = torch.stack(ref_image_list)
            gen_ref = gen_ref.to(
                memory_format=torch.contiguous_format).float()

            if cfg.show_box_on_img or cfg.show_map_on_img:
                # save ori with box
                if ori_imgs_wb is not None:
                    ori_img_with_box = output_func(ori_imgs_wb)
                    ori_img_with_box.save(
                        os.path.join(
                            cfg.log_root,
                            "frames",
                            f"{curr_index}_{batch_img_index}_ori_box.png",
                        )
                    )
                # save gen with box
                for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                    gen_img_with_box = output_func(gen_imgs_wb)
                    gen_img_with_box.save(
                        os.path.join(
                            cfg.log_root,
                            "frames",
                            f"{curr_index}_{batch_img_index}_gen{ti}_box.png",
                        )
                    )

            batch_img_index += 1
        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
