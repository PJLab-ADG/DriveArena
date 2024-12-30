import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm

import torch

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from dreamforge.runner.utils import concat_6_views
from dreamforge.misc.test_utils import (
    prepare_all, run_one_batch
)

transparent_bg = False
target_map_size = 400
# target_map_size = 800


def output_func(x): return concat_6_views(x)
# def output_func(x): return concat_6_views(x, oneline=True)
# def output_func(x): return img_concat_h(*x[:3])


@hydra.main(version_base=None, config_path="../configs",
            config_name="test_config_single")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
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
    total_num = 0
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    if not cfg.get('gen_train', False):
        os.makedirs(os.path.join(cfg.log_root, 'ori'), exist_ok=True)
    os.makedirs(os.path.join(cfg.log_root, 'gen'), exist_ok=True)
    for val_input in val_dataloader:
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=transparent_bg,
                                      map_size=target_map_size)

        batch_paths = [val_input['meta_data']['metas'][cn].data['filename'] for cn in range(len(val_input['meta_data']['metas']))]

        for i, (map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list) in enumerate(zip(*return_tuples)):
            
            paths = batch_paths[i]
            for x_img, path in zip(ori_imgs, paths):
                os.makedirs(os.path.join(cfg.log_root, 'ori', path.split('/')[-2]), exist_ok=True)
                dst_path = os.path.join(cfg.log_root, 'ori', path.split('/')[-2], path.split('/')[-1])
                x_img.save(dst_path)
            for ti, gen_imgs in enumerate(gen_imgs_list):
                for x_img, path in zip(gen_imgs, paths):
                    os.makedirs(os.path.join(cfg.log_root, 'gen', f'{ti}', path.split('/')[-2]), exist_ok=True)
                    dst_path = os.path.join(cfg.log_root, 'gen', f'{ti}', path.split('/')[-2], path.split('/')[-1])
                    x_img.save(dst_path)

            # # save map
            # # map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))

            # # save ori
            # if ori_imgs is not None:
            #     ori_img = output_func(ori_imgs)
            #     ori_img.save(os.path.join(cfg.log_root, 'ori', f"{total_num}.png"))
            # # save gen
            # for ti, gen_imgs in enumerate(gen_imgs_list):
            #     gen_img = output_func(gen_imgs)
            #     os.makedirs(os.path.join(cfg.log_root, 'gen', f'{ti}'), exist_ok=True)
            #     gen_img.save(os.path.join(
            #         cfg.log_root, 'gen', f'{ti}', f"{total_num}.png"))
            # if cfg.show_box:
            #     # save ori with box
            #     if ori_imgs_wb is not None:
            #         ori_img_with_box = output_func(ori_imgs_wb)
            #         ori_img_with_box.save(os.path.join(
            #             cfg.log_root, 'ori', f"{total_num}_box.png"))
            #     # save gen with box
            #     for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
            #         gen_img_with_box = output_func(gen_imgs_wb)
            #         gen_img_with_box.save(os.path.join(
            #             cfg.log_root, 'gen', f'{ti}', f"{total_num}_box.png"))

            total_num += 1

        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()