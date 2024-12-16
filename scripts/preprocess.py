import pdb
import os
import glob

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mpl_toolkits.axes_grid1 import make_axes_locatable

from multiprocessing import Process, Queue
import torch.multiprocessing as mp



def cotracker_worker(cotracker, frames, grid_size, queue, num_processes :int):
    """
    マルチプロセス用のCoTracker処理。
    """
    print("CoTracker worker", num_processes, "started.")
    B, T, C, H, W = frames.shape
    pred_tracks, vis_tracks = cotracker(frames, grid_size=grid_size)
    pred_tracks[..., 0] = pred_tracks[..., 0] / (W - 1) * 2 - 1
    pred_tracks[..., 1] = pred_tracks[..., 1] / (H - 1) * 2 - 1
    pred_tracks = pred_tracks.reshape(1, T, grid_size, grid_size, 2)
    vis_tracks = vis_tracks.reshape(1, T, grid_size, grid_size, 1)
    queue.put((pred_tracks, vis_tracks))
    print("finished.")


class PreProcessing():
    def __init__(self, grid_size: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature extractor (ResNet50)
        resnet = resnet50(weights="IMAGENET1K_V1").to(self.device)
        resnet.eval()
        self.layer1 = nn.Sequential(
            *list(resnet.children())[:5]  # Conv1 and Layer1
        ).to(self.device)
        self.layer2 = resnet.layer2.to(self.device)
        self.layer3 = resnet.layer3.to(self.device)
        self.layer4 = resnet.layer4.to(self.device)
        
        # CoTracker model
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        self.grid_size = grid_size
        print("grid size:", grid_size)
        
        # Image transforms
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def tracking(self, frames: torch.Tensor) -> tuple:
        assert len(frames.shape) == 5
        assert frames.max() > 1.0
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device)
        
        # 並列化用のキュー
        manager = mp.Manager()
        queue = manager.Queue()
        
        # プロセス分割
        num_processes = min(mp.cpu_count(), B)  # プロセス数をバッチサイズまたはCPUコア数に制限
        batch_split = torch.chunk(frames, num_processes, dim=0)
        processes = []
        
        # batch ごとの処理を並列化
        for i, batch in enumerate(batch_split):
            p = Process(target=cotracker_worker, args=(self.cotracker, batch, self.grid_size, queue, i))
            processes.append(p)
            p.start()
        
        # 結果を収集
        pred_tracks_list = []
        vis_tracks_list = []
        for _ in processes:
            pred_tracks, vis_tracks = queue.get()
            pred_tracks_list.append(pred_tracks)
            vis_tracks_list.append(vis_tracks)
        
        # プロセス終了
        for p in processes:
            p.join()
        
        # 結果を結合
        pred_tracks = torch.cat(pred_tracks_list, dim=0)
        vis_tracks = torch.cat(vis_tracks_list, dim=0)
        
        return pred_tracks, vis_tracks

    def extract_featmaps(self, img: torch.Tensor) -> tuple:
        _ = self.img_transform(img / 255.0)
        layer1 = self.layer1(_)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return (layer4, layer3, layer2, layer1)
    
    def feature_pyramid_extract(self, frames: torch.Tensor) -> tuple:
        assert len(frames.shape) == 5
        assert frames.shape[1] == 2
        B, T, C, H, W = frames.shape
        
        tgt_featmaps = self.extract_featmaps(frames[:, 0, :, :, :])
        supp_featmaps = self.extract_featmaps(frames[:, 1, :, :, :])
        
        pred_tracks, vis_tracks = self.tracking(frames)
        
        tgt_output = []
        supp_output = []
        
        for i in range(4):
            tgt_output.append(F.grid_sample(tgt_featmaps[i], pred_tracks[:, 0], align_corners=True))
            supp_output.append(F.grid_sample(supp_featmaps[i], pred_tracks[:, 1], align_corners=True))
        
        tgt_output = torch.cat(tgt_output, dim=1)
        supp_output = torch.cat(supp_output, dim=1)
        
        return tgt_output, supp_output
