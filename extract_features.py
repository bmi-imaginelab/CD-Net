# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

input_image_size = 224
# python -m torch.distributed.launch --nproc_per_node=3 extract_features.py

def extract_feature_pipeline(args):
    # ============ preparing data ... ============

    dataset_train = Dataset(args.data_path + '/train_5x_list.pickle')
    dataset_val = Dataset(args.data_path + '/test_5x_list.pickle')

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)

    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)

    # save features
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
    
    del train_features
    
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    # save features
    if args.dump_features and dist.get_rank() == 0:
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        
    # return train_features, test_features


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
           
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

class Dataset():
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.files_list = pickle.load(f)

        self.to_tensor = albumentations.Compose(
            [
                ToTensorV2()
            ],
        )
     
    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        
        img = cv2.cvtColor(
            cv2.imread(temp_path)[:,:,:3], cv2.COLOR_BGR2RGB).astype(np.float32)/255.0   # albumentations
                
        return (self.to_tensor(image=img)['image'], []), idx

###############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting features for downstream tasks')
    parser.add_argument('--batch_size_per_gpu', default=100, type=int, help='Per-GPU batch-size')
    
    parser.add_argument('--pretrained_weights', default='.../Experiments/Lung_cancer/DINO_5X/vit_small_fp16true_momentum996_outdim65536/checkpoint0099.pth', type=str, help="Path to pretrained weights to evaluate.")
    
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    
    parser.add_argument('--dump_features', default='.../Experiments/Lung_cancer/DINO_5X/vit_small_fp16true_momentum996_outdim65536/cls_token_features_ep99',
        help='Path where to save computed features, empty for no saving')
    
    
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default='tcp://localhost:10002', type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    parser.add_argument('--data_path', default='.../Datasets/Lung_cancer/train_test_split', type=str)
    
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    extract_feature_pipeline(args)


