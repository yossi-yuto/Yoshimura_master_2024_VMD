import numpy as np
import os
import argparse
import importlib
import pdb
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import VMD_test_root
from misc import check_mkdir
# from networks.TVSD import TVSD

from dataset.VShadow_crosspairwise_proposed import listdirs_only
import argparse
from tqdm import tqdm
from preprocess import PreProcessing
from metrics import calc_iou_multi_thresh, get_maxFscore_and_threshold_multi_beta
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='proposed_network', help='model name')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()

args_ = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args_.gpu

net_file = importlib.import_module('networks.' + args_.model)
net = net_file.VMD_Network().cuda()

args = {
    'scale': 416,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'SegmentationClassPNG'
}

# PreProcess function   
img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor()
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()
preprocess = PreProcessing(grid_size=52)

root = VMD_test_root[0]


def main():
    
    checkpoint = args_.param_path
    check_point = torch.load(checkpoint, weights_only=True)
    net.load_state_dict(check_point['model'])

    net.eval()
    with torch.no_grad():
        video_list = listdirs_only(os.path.join(root))
        
        metrics_dict = {
            'IoU03': [],
            'IoU05': [],
            'IoU08': [],
            'MAE': [],
            'MaxFbeta0.3': [],
            'MaxFbeta0.5': [],
            'MaxFbeta1.0': [],
        }
        
        for video in tqdm(video_list):
            # all images
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['input_folder'],)) if
                        f.endswith('.jpg')]
            # need evaluation images
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['label_folder'])) if f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)
            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
                query_idx_list = getAdjacentIndex(exemplar_idx, 0, len(img_list), args['test_adjacent']) # インデックスは一つのみ返す
                other_idx_list = getNonAdjacentRandomIndex(current_index=exemplar_idx, video_length=len(img_list), min_distance=4, num_samples=1) # インデックスは一つのみ返す
                
                for query_idx in query_idx_list:  
                    exemplar = Image.open(os.path.join(root, video, args['input_folder'], exemplar_name + '.jpg')).convert('RGB')
                    exemplar_gt = Image.open(os.path.join(root, video, args['label_folder'], exemplar_name + '.png')).convert('1')
                    w, h = exemplar.size
                    query = Image.open(os.path.join(root, video, args['input_folder'], img_list[query_idx] + '.jpg')).convert('RGB')
                    other = Image.open(os.path.join(root, video, args['input_folder'], img_list[other_idx_list[0]] + '.jpg')).convert('RGB')
                    exemplar_tensor = img_transform(exemplar).unsqueeze(0).cuda()
                    query_tensor = img_transform(query).unsqueeze(0).cuda()
                    other_tensor = img_transform(other).unsqueeze(0).cuda()
                    print("exemplar_name: ", os.path.join(root, video, args['input_folder'], exemplar_name + '.jpg'), 
                          "\nquery_name: ", os.path.join(root, video, args['input_folder'], img_list[query_idx] + '.jpg'),
                            "\nother_name: ", os.path.join(root, video, args['input_folder'], img_list[other_idx_list[0]] + '.jpg'))
                    # proposed
                    frames = torch.stack([mask_transform(exemplar), mask_transform(other)], dim=0)
                    frames = (frames * 255.).unsqueeze(0).cuda()
                    output = preprocess.feature_pyramid_extract(frames)
                    
                    exemplar_pre, _, _, opflow_based_map = net(exemplar_tensor, query_tensor, other_tensor, output['exempler_featmap'], output['query_featmap'], output['opflow_angle'], output['opflow_magnitude'])
                    res = (exemplar_pre.data > 0).to(torch.float32).squeeze(0)
                    res_sigmoid = torch.sigmoid(exemplar_pre) # 0~1
                    
                    pred_1d = res_sigmoid.cpu().numpy().flatten()
                    gt_1d = mask_transform(exemplar_gt).numpy().flatten()
                    # calculate IoU
                    # pdb.set_trace()
                    IoU :dict  = calc_iou_multi_thresh(gt_1d, pred_1d, thresholds=[0.3, 0.5, 0.8])
                    metrics_dict['IoU03'].append(IoU[0.3])
                    metrics_dict['IoU05'].append(IoU[0.5])
                    metrics_dict['IoU08'].append(IoU[0.8])
                    metrics_dict['IoU05'].append(IoU)
                    # calculate MAE
                    MAE = np.mean(np.abs(gt_1d - pred_1d))
                    metrics_dict['MAE'].append(MAE)
                    # calculate maximum Fbeta
                    fbetas :dict = get_maxFscore_and_threshold_multi_beta(gt_1d, pred_1d, betas=[0.3, 0.5, 1.0])
                    metrics_dict['MaxFbeta0.3'].append(fbetas[0.3]['max_fscore'])
                    metrics_dict['MaxFbeta0.5'].append(fbetas[0.5]['max_fscore'])
                    metrics_dict['MaxFbeta1.0'].append(fbetas[1.0]['max_fscore'])
                    
                    print(" IoU05 : ", round(IoU, 4), 
                        "\nMAE : ", round(MAE, 4), 
                        "\nMaxmumFbeta0.3: ", round(fbetas[0.3]['max_fscore'], 4),
                        "\nMaxmumFbeta0.5: ", round(fbetas[0.5]['max_fscore'], 4),
                        "\nMaxmumFbeta1.0: ", round(fbetas[1.0]['max_fscore'], 4))

                    
                    check_mkdir(os.path.join(args_.result_path, video))
                    save_name = f"{exemplar_name}.png"
                    print(os.path.join(args_.result_path, video, save_name))

                    """ optical flow based map visualization """
                    opflow_based_map = F.interpolate(torch.sigmoid(opflow_based_map), size=(h, w), mode='bilinear', align_corners=False)
                    
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                    axes[0, 0].imshow(exemplar, cmap='gray')  # グレースケール画像の場合は cmap='gray'
                    axes[0, 0].set_title('Target image')
                    axes[0, 0].axis('off')

                    axes[0, 1].imshow(exemplar_gt, cmap='gray')
                    axes[0, 1].set_title('Ground truth')
                    axes[0, 1].axis('off')

                    resized_sigmoid = F.interpolate(res_sigmoid, size=(h, w), mode='bilinear', align_corners=False)
                    axes[1, 0].imshow(resized_sigmoid.squeeze().cpu().numpy(), vmin=0, vmax=1, cmap='gray')
                    axes[1, 0].set_title('Final prediction')
                    axes[1, 0].axis('off')

                    opflow_based_pred = opflow_based_map.squeeze().cpu().numpy()
                    axes[1, 1].imshow(opflow_based_map.squeeze().cpu().numpy(), vmin=0, vmax=1, cmap='gray')
                    axes[1, 1].set_title('Optical flow-based map')
                    axes[1, 1].axis('off')

                    # 保存
                    Image.fromarray((opflow_based_pred * 255).astype(np.uint8)).save(os.path.join(args_.result_path, video, f"{exemplar_name}_opflow_based.png"))
                    result_file = os.path.join(args_.result_path, video, f"{exemplar_name}_result.png")
                    print(result_file)
                    os.makedirs(os.path.dirname(result_file), exist_ok=True)  # 保存ディレクトリを作成
                    fig.savefig(result_file)

                    # リソース解放
                    plt.close(fig)
                    
                    """ maxFbeta threshold visualization """
                    thres = fbetas[0.3]['threshold']
                    score = fbetas[0.3]['max_fscore']
                    maxFbeta_map = torch.zeros_like(resized_sigmoid)
                    maxFbeta_map[resized_sigmoid> thres] = 255
                    Image.fromarray((maxFbeta_map.squeeze().cpu().numpy()).astype(np.uint8)).save(os.path.join(args_.result_path, video, f"{exemplar_name}_maxFbeta_{score:.3F}.png"))
                    continue

        with open(os.path.join(args_.result_path, 'metrics.txt'), 'a') as f:
            f.write("IoU03 : {}\n".format(np.mean(metrics_dict['IoU03'])))
            f.write("IoU05 : {}\n".format(np.mean(metrics_dict['IoU05'])))
            f.write("IoU08 : {}\n".format(np.mean(metrics_dict['IoU08'])))
            f.write("MAE : {}\n".format(np.mean(metrics_dict['MAE'])))
            f.write("MaxFbeta0.3 : {}\n".format(np.mean(metrics_dict['MaxFbeta0.3'])))
            f.write("MaxFbeta0.5 : {}\n".format(np.mean(metrics_dict['MaxFbeta0.5'])))
            f.write("MaxFbeta1.0 : {}\n".format(np.mean(metrics_dict['MaxFbeta1.0'])))

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



def getNonAdjacentRandomIndex(current_index, video_length, min_distance, num_samples=1):
    """
    Get random indices from the same video that are not adjacent and are at least
    `min_distance` frames away from the current index.

    Args:
        current_index (int): The index of the current frame.
        video_length (int): The total number of frames in the video.
        min_distance (int): The minimum distance from the current index.
        num_samples (int): The number of random indices to return.

    Returns:
        list: A list of random indices that meet the criteria.
    """
    # Create a list of valid indices
    # pdb.set_trace()
    # dis_list = [abs(i - current_index) for i in range(video_length)]
    # max_idx = dis_list.index(max(dis_list))
    # valid_indices = [max_idx]
    valid_indices = [i for i in range(video_length) if abs(i - current_index) >= min_distance]

    # Randomly sample the required number of indices
    if len(valid_indices) < num_samples:
        raise ValueError("Not enough valid frames to sample the requested number of indices.")

    return random.sample(valid_indices, num_samples)

if __name__ == '__main__':
    main()
