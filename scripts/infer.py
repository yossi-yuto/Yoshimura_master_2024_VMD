import numpy as np
import os
import pdb

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from config import VMD_test_root
from misc import check_mkdir
from networks.VMD_network import VMD_Network
from dataset.VShadow_crosspairwise import listdirs_only
from tqdm import tqdm
from metrics import calc_iou_multi_thresh, get_maxFscore_and_threshold_multi_beta
from infer_proposed import parse_args


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
mask_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor()
])

target_transform = transforms.ToTensor()

root = VMD_test_root[0]

to_pil = transforms.ToPILImage()


def main():
    net = VMD_Network().cuda()
    # pdb.set_trace()
    opt = parse_args()
    checkpoint = os.path.join(opt.result_path, 'best_mae.pth')
    check_point = torch.load(checkpoint)
    net.load_state_dict(check_point['model'])
    
    ''' 結果を保存するディレクトリ '''
    result_dir = os.path.join(opt.result_path, 'test')
    os.makedirs(result_dir, exist_ok=True)
    
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
            
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['input_folder'],)) if
                        f.endswith('.jpg')]
            
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['label_folder'])) if f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)
            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
                query_idx_list = getAdjacentIndex(exemplar_idx, 0, len(img_list), args['test_adjacent'])

                for query_idx in query_idx_list:
                    exemplar = Image.open(os.path.join(root, video, args['input_folder'], exemplar_name + '.jpg')).convert('RGB')
                    exemplar_gt = Image.open(os.path.join(root, video, args['label_folder'], exemplar_name + '.png')).convert('1')
                    w, h = exemplar.size
                    query = Image.open(os.path.join(root, video, args['input_folder'], img_list[query_idx] + '.jpg')).convert('RGB')
                    exemplar_tensor = img_transform(exemplar).unsqueeze(0).cuda()
                    query_tensor = img_transform(query).unsqueeze(0).cuda()
                    exemplar_pre, _, _ = net(exemplar_tensor, query_tensor, query_tensor)
                    res = (exemplar_pre.data > 0).to(torch.float32).squeeze(0)
                    res_sigmoid = torch.sigmoid(exemplar_pre.squeeze()) # 0~1
                    
                    pred_1d = res_sigmoid.cpu().numpy().flatten()
                    gt_1d = mask_transform(exemplar_gt).numpy().flatten()
                    # calculate IoU
                    IoU :dict  = calc_iou_multi_thresh(gt_1d, pred_1d,  thresholds=[0.3, 0.5, 0.8])
                    metrics_dict['IoU03'].append(IoU[0.3])
                    metrics_dict['IoU05'].append(IoU[0.5])
                    metrics_dict['IoU08'].append(IoU[0.8])
                    # calculate MAE
                    MAE: float = np.mean(np.abs(gt_1d - pred_1d))
                    metrics_dict['MAE'].append(MAE)
                    # calculate maximum Fbeta
                    fbetas :dict = get_maxFscore_and_threshold_multi_beta(gt_1d, pred_1d, betas=[0.3, 0.5, 1.0])
                    metrics_dict['MaxFbeta0.3'].append(fbetas[0.3]['max_fscore'])
                    metrics_dict['MaxFbeta0.5'].append(fbetas[0.5]['max_fscore'])
                    metrics_dict['MaxFbeta1.0'].append(fbetas[1.0]['max_fscore'])
                    
                    print(" IoU05 : ", IoU, 
                        "\nMAE : ", MAE, 
                        "\nMaxmumFbeta0.3: ", fbetas[0.5]['max_fscore'],
                        "\nMaxmumFbeta0.5: ", fbetas[0.5]['max_fscore'],
                        "\nMaxmumFbeta1.0: ", fbetas[1.0]['max_fscore'])
                    check_mkdir(os.path.join(result_dir, video))
                    
                    thres = fbetas[0.3]['threshold']
                    score = fbetas[0.3]['max_fscore']
                    maxFbeta_map = torch.zeros_like(res_sigmoid)
                    maxFbeta_map[res_sigmoid > thres] = 1.0
                    maxFbeta_map = F.interpolate(maxFbeta_map[None,None], size=(h, w), mode='bilinear', align_corners=False)
                    Image.fromarray((maxFbeta_map.squeeze().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(result_dir, video, f"{exemplar_name}_maxFbeta_{score:.3f}.png"))
            
        with open(os.path.join(result_dir, 'metrics.txt'), 'a') as f:
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

if __name__ == '__main__':
    main()
