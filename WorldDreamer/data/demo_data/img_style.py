import cv2
import os.path as osp
import numpy as np
import torch
from torchvision import transforms

transform1 = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean= [0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5]),
    ])

def style_dict(style, dataset_root):
    style_dict = {
        'boston': [
            'samples/CAM_FRONT_LEFT/n008-2018-08-31-11-37-23-0400__CAM_FRONT_LEFT__1535729899904799.jpg',
            'samples/CAM_FRONT/n008-2018-08-31-11-37-23-0400__CAM_FRONT__1535729899912404.jpg',
            'samples/CAM_FRONT_RIGHT/n008-2018-08-31-11-37-23-0400__CAM_FRONT_RIGHT__1535729899920482.jpg',
            'samples/CAM_BACK_RIGHT/n008-2018-08-31-11-37-23-0400__CAM_BACK_RIGHT__1535729899928113.jpg',
            'samples/CAM_BACK/n008-2018-08-31-11-37-23-0400__CAM_BACK__1535729899937558.jpg',
            'samples/CAM_BACK_LEFT/n008-2018-08-31-11-37-23-0400__CAM_BACK_LEFT__1535729899947405.jpg',
        ],
        'boston_rain':[
            'samples/CAM_FRONT_LEFT/n008-2018-09-18-14-54-39-0400__CAM_FRONT_LEFT__1537297192854799.jpg',
            'samples/CAM_FRONT/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297192862404.jpg',
            'samples/CAM_FRONT_RIGHT/n008-2018-09-18-14-54-39-0400__CAM_FRONT_RIGHT__1537297192870482.jpg',
            'samples/CAM_BACK_RIGHT/n008-2018-09-18-14-54-39-0400__CAM_BACK_RIGHT__1537297192878113.jpg',
            'samples/CAM_BACK/n008-2018-09-18-14-54-39-0400__CAM_BACK__1537297192887558.jpg',
            'samples/CAM_BACK_LEFT/n008-2018-09-18-14-54-39-0400__CAM_BACK_LEFT__1537297192897405.jpg',
        ],
        'boston_sunny':[
            'samples/CAM_FRONT_LEFT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639644154799.jpg',
            'samples/CAM_FRONT/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639644162881.jpg',
            'samples/CAM_FRONT_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_RIGHT__1535639644170482.jpg',
            'samples/CAM_BACK_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_BACK_RIGHT__1535639644178113.jpg',
            'samples/CAM_BACK/n008-2018-08-30-10-33-52-0400__CAM_BACK__1535639644187558.jpg',
            'samples/CAM_BACK_LEFT/n008-2018-08-30-10-33-52-0400__CAM_BACK_LEFT__1535639644197405.jpg',
        ],
        'hollandvillage_night':[
            'samples/CAM_FRONT_LEFT/n015-2018-11-21-19-21-35+0800__CAM_FRONT_LEFT__1542799563354844.jpg',
            'samples/CAM_FRONT/n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799563362460.jpg',
            'samples/CAM_FRONT_RIGHT/n015-2018-11-21-19-21-35+0800__CAM_FRONT_RIGHT__1542799563370339.jpg',
            'samples/CAM_BACK_RIGHT/n015-2018-11-21-19-21-35+0800__CAM_BACK_RIGHT__1542799563377893.jpg',
            'samples/CAM_BACK/n015-2018-11-21-19-21-35+0800__CAM_BACK__1542799563387525.jpg',
            'samples/CAM_BACK_LEFT/n015-2018-11-21-19-21-35+0800__CAM_BACK_LEFT__1542799563397423.jpg'
        ]
    }
    ref_images = []
    if style in style_dict:
        img_path = style_dict[style]
        for path in img_path:
            tmp = cv2.imread(osp.join(dataset_root, path))
            tmp_rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp_rgb = cv2.resize(tmp_rgb, (400, 224))
            tmp_rgb = transform1(tmp_rgb)
            ref_images.append(tmp_rgb)
    else:
        ref_images = np.ones((224,400,3))*255
        ref_images = transform1(ref_images)
        ref_images = ref_images.unsqueeze(0).repeat(6, 1, 1, 1)
    ref_images = torch.stack(ref_images)
    ref_images = ref_images.cuda()
    return ref_images