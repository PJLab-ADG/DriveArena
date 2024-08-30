from fastapi import FastAPI, Request, BackgroundTasks
import base64
from PIL import Image
import io
import copy
import os
import uvicorn
from argparse import ArgumentParser
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
import sys
from hydra import initialize, compose
from omegaconf import OmegaConf
from functools import partial

sys.path.append(".")  # noqa
from projects.dreamer.utils.test_utils import (
    run_one_batch, build_pipe, update_progress_bar_config, collate_fn_singleframe
)
from torchvision import transforms
from dataset import ApiSetWrapper
import torch
from fastapi.responses import StreamingResponse  
from io import BytesIO  
from projects.dreamer.runner.utils import concat_6_views
import numpy as np
from typing import Dict
from queue import Queue  
import json
from pydantic import BaseModel
import requests
from data.demo_data.img_style import style_dict
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

transform1 = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5]),
    ])

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--tmp-dir", type=str, default='tmp')
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--agent_port", type=int, default=11001)
    parser.add_argument("--resume", type=str, default='dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000')

    args = parser.parse_args()
    return args


app = FastAPI()
args = _get_args()

batch_index = 0
gen_ref = None
transparent_bg = False
target_map_size = 400
timestamp = 0.
agent_command = 2
def output_func(x): return concat_6_views(x, oneline=True)

class Data(BaseModel):
    task_id: str
    data: Dict


@app.post("/dreamer-api/")
async def process(request: Request, background_tasks: BackgroundTasks):
    param = await request.json()
    if isinstance(param, str):
        param = json.loads(param)
    global batch_index, gen_ref, agent_command
    val_dataloader = load_data(cfg, pipe, param)
    agent_command = param['agent_command']

    for val_input in val_dataloader:
        batch_index += 1

        if gen_ref is None:
            ref_images = style_dict('boston', cfg.dataset.dataset_root_nuscenes)
            val_input['ref_images'][0, ...] = ref_images
            val_input['relative_pose'][0] = torch.eye(4)
        else:
            val_input["ref_images"][0,...]= gen_ref

        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=transparent_bg,
                                      map_size=target_map_size)
        
        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            gen_imgs = gen_imgs_wb_list[0]
            ref_image_list = []
            for cam_i in range(6):
                img_i = gen_imgs[cam_i]
                img_i = np.array(img_i)
                img_i = transform1(img_i)
                ref_image_list.append(img_i)
            gen_ref = torch.stack(ref_image_list)         
            gen_ref = gen_ref.to(memory_format=torch.contiguous_format).float()
    
    combined_image = Image.new("RGB", (gen_imgs[0].width, sum(img.height for img in gen_imgs)))  
    y_offset = 0  
    for img in gen_imgs:  
        combined_image.paste(img, (0, y_offset))  
        y_offset += img.height  
    img_byte_array = BytesIO()  
    combined_image.save(img_byte_array, format="PNG")  
    img_byte_array.seek(0)
    # add into image_queue
    image_queue.put({'param': param, 'img': img_byte_array})
    background_tasks.add_task(send2agent)

    return StreamingResponse(io.BytesIO(img_byte_array.read()), media_type="image/png")  


async def send2agent():
    global timestamp, agent_command, args
    print("current timestamp:", timestamp)
    if image_queue.qsize() >= 1:
        cur_data = image_queue.get() 
        img_byte_array = cur_data['img']
        img_byte_array = img_byte_array.getvalue()
        image_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        ego_pose = cur_data['param']['metas']['ego_pos']
        accel = cur_data['param']['metas']['accel']
        rotation_rate = cur_data['param']['metas']['rotation_rate']
        vel = cur_data['param']['metas']['vel']
        send_data = {
            'timestamp':timestamp, 
            'img_byte_array': image_base64, 
            'ego_pose':ego_pose, 
            'command': agent_command, 
            'accel': accel, 
            'rotation_rate': rotation_rate, 
            'vel': vel}
        response = requests.post("http://localhost:{}/driver-api".format(args.agent_port), json=send_data)
        timestamp = timestamp + 0.5


@app.get("/dreamer-clean/")
async def clean_history():
    print('[world dreamer] cleaned out past frames')
    global gen_ref, timestamp
    gen_ref = None
    timestamp = 0.
    while image_queue.qsize() > 0:  
        image_queue.get()  


def load_model(cfg, device='cuda'):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    return  pipe, weight_dtype

    
def load_data(cfg, pipe, param):
    val_dataset = ApiSetWrapper(param)
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

    def _collate_fn(examples, *args, **kwargs):
        return collate_fn_singleframe(examples, *args, **kwargs)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(_collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return val_dataloader


if __name__ == '__main__':
    os.makedirs(args.tmp_dir, exist_ok=True)
    image_queue = Queue()  
    img_pad_size = [928, 1600]

    initialize(version_base=None, config_path="../configs")
    cfg = compose("test_config")
    cfg.resume_from_checkpoint = args.resume
    pipe, weight_dtype= load_model(cfg)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")



