import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

from config import VMD_test_root
from misc import check_mkdir
# from networks.TVSD import TVSD
from networks.VMD_network import VMD_Network
from dataset.VShadow_crosspairwise import listdirs_only
import argparse
from tqdm import tqdm
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = {
    'scale': 416,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'SegmentationClassPNG'
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

root = VMD_test_root[0]

to_pil = transforms.ToPILImage()


def main():
    net = VMD_Network().cuda()

    
    checkpoint = 'best.pth'
    check_point = torch.load(checkpoint)
    net.load_state_dict(check_point['model'])

    net.eval()
    with torch.no_grad():
        video_list = listdirs_only(os.path.join(root))
        for video in tqdm(video_list):
            # all images
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['input_folder'],)) if
                        f.endswith('.jpg')]
            # need evaluation images
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['label_folder'])) if f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)
            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
                query_idx_list = getAdjacentIndex(exemplar_idx, 0, len(img_list), args['test_adjacent'])

                for query_idx in query_idx_list:
                    exemplar = Image.open(os.path.join(root, video, args['input_folder'], exemplar_name + '.jpg')).convert('RGB')
                    w, h = exemplar.size
                    query = Image.open(os.path.join(root, video, args['input_folder'], img_list[query_idx] + '.jpg')).convert('RGB')
                    exemplar_tensor = img_transform(exemplar).unsqueeze(0).cuda()
                    query_tensor = img_transform(query).unsqueeze(0).cuda()
                    exemplar_pre, _, _ = net(exemplar_tensor, query_tensor, query_tensor)
                    res = (exemplar_pre.data > 0).to(torch.float32).squeeze(0)
                    # res = torch.sigmoid(exemplar_pre.squeeze())
                    prediction = np.array(
                        transforms.Resize((h, w))(to_pil(res.cpu())))

                    check_mkdir(os.path.join("results", video))
                    save_name = f"{exemplar_name}.png"
                    print(os.path.join("results", video, save_name))
                    Image.fromarray(prediction).save(os.path.join("results", video, save_name))


def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def getAdjacentIndex(current_index, start_index, video_length, adjacent_length):
    if current_index + adjacent_length < start_index + video_length:
        query_index_list = [current_index+i+1 for i in range(adjacent_length)]
    else:
        query_index_list = [current_index-i-1 for i in range(adjacent_length)]
    return query_index_list

if __name__ == '__main__':
    main()
