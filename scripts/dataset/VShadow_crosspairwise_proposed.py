import os
import os.path
import pdb

import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

class CrossPairwiseImg(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
        self.img_root, self.video_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        self.videoImg_list = self.generateImgFromVideo(self.video_root) # list of (img_path, gt_path, videoStartIndex, videoLength)
        
        print('Total video frames is {}.'.format(self.num_video_frame))
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))

    def __getitem__(self, index):
        # 検出対象の画像を取得（連続するフレームID）
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]
        
        # 検出対象の画像における動画データの情報を取得
        frame_index_list = np.arange(videoStartIndex, videoStartIndex + videoLength) 
        
        # 次のフレームが存在しない場合は、前のフレームを参照
        if index == frame_index_list[-1]:
            query_index = index - 1
        else:
            query_index = index + 1
        query_path, query_gt_path, _, _ = self.videoImg_list[query_index]
        
        # 検出対象フレームIDと動画内の各フレームIDの距離を計算
        relative_dis_index = frame_index_list - index 
        
        """ 最も離れたフレームを取得 """
        relative_max_far_index = max(relative_dis_index, key=abs)
        other_index = index + relative_max_far_index
        
        """ 5フレーム間を設定 """
        # if -5 in relative_dis_index:
        #     query_index = index - 5
        # elif 5 in relative_dis_index:
        #     query_index = index + 5
        # else:
        #     max_relative_index = max(relative_dis_index, key=abs)
        #     query_index = index + max_relative_index
        
        other_path, other_gt_path, _, _ = self.videoImg_list[other_index]

        exemplar = Image.open(exemplar_path).convert('RGB')
        query = Image.open(query_path).convert('RGB')
        other = Image.open(other_path).convert('RGB')
        exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        query_gt = Image.open(query_gt_path).convert('L')
        other_gt = Image.open(other_gt_path).convert('L')

        # データ拡張
        manual_random = random.random()
        if self.joint_transform is not None:
            exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
            query, query_gt = self.joint_transform(query, query_gt, manual_random)
            other, other_gt = self.joint_transform(other, other_gt)
            if len(self.img_root) > 0:
                single_image, single_gt = self.joint_transform(single_image, single_gt)
            exemplar_frame = self.target_transform(exemplar) * 255.0
            other_frame = self.target_transform(other) * 255.0
            frames = torch.stack([exemplar_frame, other_frame], dim=0) # (2, C, H, W)
        # 通常の変換
        if self.img_transform is not None:
            exemplar = self.img_transform(exemplar)
            query = self.img_transform(query)
            other = self.img_transform(other)
            if len(self.img_root) > 0:
                single_image = self.img_transform(single_image)
        # マスク画像の変換
        if self.target_transform is not None:
            exemplar_gt = self.target_transform(exemplar_gt)
            query_gt = self.target_transform(query_gt)
            other_gt = self.target_transform(other_gt)
            if len(self.img_root) > 0:
                single_gt = self.target_transform(single_gt)
                
        if len(self.img_root) > 0:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'query': query, 'query_gt': query_gt,
                      'other': other, 'other_gt': other_gt, 'single_image': single_image, 'single_gt': single_gt, 'frames': frames}
        else:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'query': query, 'query_gt': query_gt,
                      'other': other, 'other_gt': other_gt, 'frames': frames}
        return sample

    def generateImgFromVideo(self, root):
        imgs = []
        root = root[0]
        video_list = listdirs_only(os.path.join(root[0]))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], video, self.input_folder)) if f.endswith(self.img_ext)]
            img_list = self.sortImg(img_list)
            for img in img_list:
                videoImgGt = (os.path.join(root[0], video, self.input_folder, img + self.img_ext),
                              os.path.join(root[0], video, self.label_folder, img + self.label_ext), self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)
        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))
        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
                for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list) // 2 * 2
    

if __name__ == '__main__':
    root = [
        ("/path/to/image_dataset", "image", "ImageSet1"),
        ("/path/to/video_dataset", "video", "VideoSet1")
    ]

    joint_transform = None
    img_transform = None
    target_transform = None

    dataset = CrossPairwiseImg(
        root=root, 
        joint_transform=joint_transform, 
        img_transform=img_transform,
        target_transform=target_transform
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        print("Index:", i)
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
        if i == 2:
            break
