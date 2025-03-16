import datetime
import os
import pdb

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from config import VMD_training_root, VMD_valid_root
from dataset.VShadow_crosspairwise_proposed import CrossPairwiseImg
from misc import AvgMeter, check_mkdir
import math
from losses import lovasz_hinge, binary_xloss
import torch.nn.functional as F
import numpy as np
import time
import argparse
import importlib
from preprocess import PreProcessing


# 実験設定
cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = './experiment_results'

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='proposed_network', help='exp name')
parser.add_argument('--model', type=str, default='proposed_network', help='model name')
parser.add_argument('--gpu', type=str, default='1', help='used gpu id')
parser.add_argument('--batchsize', type=int, default=10, help='train batch')
parser.add_argument('--fold', type=int, default=0, help='fold number')
parser.add_argument('--bestonly', action="store_true", help='only best model')

cmd_args = parser.parse_args()
exp_name = cmd_args.exp
model_name = cmd_args.model
gpu_ids = cmd_args.gpu
train_batch_size = cmd_args.batchsize
fold_num = cmd_args.fold

VMD_file = importlib.import_module('networks.' + model_name)
VMD_Network = VMD_file.VMD_Network

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

args = {
    # 'exp_name': exp_name,
    'max_epoch': 15,
    # 'train_batch_size': 10,
    'last_iter': 0,
    # 'finetune_lr': 5e-5,
    'finetune_lr': 1e-4,
    # 'finetune_lr': 1e-3,
    # 'scratch_lr': 5e-4,
    'scratch_lr': 1e-3,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'multi-scale': None,
    # 'gpu': '4,5',
    # 'multi-GPUs': True,
    'fp16': False,
    'warm_up_epochs': 3,
    'seed': 2020
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

# multi-GPUs training
if len(gpu_ids.split(',')) > 1:
    batch_size = train_batch_size * len(gpu_ids.split(','))
# single-GPU training
else:
    torch.cuda.set_device(0)
    batch_size = train_batch_size

# 前処理用コード
preprocess = PreProcessing(grid_size=52, relative_flow=True)

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
VMD_training_root_list = list(VMD_training_root)
VMD_training_root_list[0] = VMD_training_root_list[0] 
VMD_valid_root_list = list(VMD_valid_root)
VMD_valid_root_list[0] = VMD_valid_root_list[0] 
training_root = [VMD_training_root_list]
train_set = CrossPairwiseImg(training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=batch_size,  drop_last=True, num_workers=0, shuffle=True)
valid_root = [VMD_valid_root_list]
val_set = CrossPairwiseImg(valid_root, val_joint_transform, img_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, shuffle=False)

print("max epoch:{}".format(args['max_epoch']))


log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
val_log_path = os.path.join(ckpt_path, exp_name, 'val_log' + str(datetime.datetime.now()) + '.txt')

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if len(gpu_ids.split(',')) > 1:
        net = torch.nn.DataParallel(VMD_Network()).cuda().train()
        for name, param in net.named_parameters():
            if 'backbone' in name:
                print(name)
        # net = net.apply(freeze_bn) # freeze BN
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name], "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name], "lr": args['scratch_lr']},
            # {"params": net.module.encoder.backbone.parameters(), "lr": args['finetune_lr']},
            # {"params": net.module.encoder.aspp.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.encoder.final_pre.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.ra_attention.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.project.parameters(), "lr": args['scratch_lr']},
            # {"params": net.module.final_pre.parameters(), "lr": args['scratch_lr']}
        ]
    # single-GPU training
    else:
        net = VMD_Network().cuda().train()
        params = [
            {"params": [param for name, param in net.named_parameters() if 'backbone' in name], "lr": args['finetune_lr']},
            {"params": [param for name, param in net.named_parameters() if 'backbone' not in name], "lr": args['scratch_lr']},
            # {"params": net.encoder.backbone.parameters(), "lr": args['finetune_lr']},
            # {"params": net.encoder.aspp.parameters(), "lr": args['scratch_lr']},
            # {"params": net.encoder.final_pre.parameters(), "lr": args['scratch_lr']},
            # {"params": net.ra_attention.parameters(), "lr": args['scratch_lr']},
            # {"params": net.project.parameters(), "lr": args['scratch_lr']},
            # {"params": net.final_pre.parameters(), "lr": args['scratch_lr']}
        ]

    for name, param in net.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False

    # learning setting
    optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
                             (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer, scheduler)


def train(net, optimizer, scheduler):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    best_mae = 100.0

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7, loss_record8 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        
        start_time = time.time()
        for i, sample in enumerate(train_iterator):
            exemplar, exemplar_gt, query, query_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda(), sample['query'].cuda(), sample['query_gt'].cuda()
            other, other_gt = sample['other'].cuda(), sample['other_gt'].cuda()   
            
            output = preprocess.feature_pyramid_extract(sample['frames'].cuda())

            optimizer.zero_grad()

            exemplar_pre, query_pre, other_pre, examplar_final, query_final, other_final, opflow_based_pred = net(exemplar, query, other, output['exempler_featmap'], output['query_featmap'], output['opflow_angle'], output['opflow_magnitude'])

            loss_hinge1 = lovasz_hinge(exemplar_pre, exemplar_gt)
            loss_hinge2 = lovasz_hinge(query_pre, query_gt)
            loss_hinge3 = lovasz_hinge(other_pre, other_gt)

            loss_hinge_examplar = lovasz_hinge(examplar_final, exemplar_gt)
            loss_hinge_query = lovasz_hinge(query_final, query_gt)
            loss_hinge_other = lovasz_hinge(other_final, other_gt)
            
            loss_hinge_opflow = lovasz_hinge(opflow_based_pred, exemplar_gt)
            
            loss_seg = loss_hinge1 + loss_hinge2 + loss_hinge3 + loss_hinge_examplar + loss_hinge_query + loss_hinge_other + loss_hinge_opflow
            loss = loss_seg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  
            optimizer.step()  

            loss_record1.update(loss_hinge_examplar.item(), batch_size)
            loss_record2.update(loss_hinge_query.item(), batch_size)
            loss_record3.update(loss_hinge_other.item(), batch_size)
            loss_record4.update(loss_hinge1.item(), batch_size)
            loss_record5.update(loss_hinge2.item(), batch_size)
            loss_record6.update(loss_hinge3.item(), batch_size)
            loss_record7.update(loss_hinge_opflow.item(), batch_size)

            curr_iter += 1

            log = "epochs:%d, iter: %d, hinge1_f: %f5, hinge2_f: %f5, hinge3_f: %f5, hinge1: %f5, hinge2: %f5, hinge3: %f5, hinge opflow: %f5, lr: %f8"%\
                  (curr_epoch, curr_iter, loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, loss_record5.avg,
                   loss_record6.avg, loss_record7.avg, scheduler.get_lr()[0])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.perf_counter() - start)
                start = time.perf_counter()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
            open(log_path, 'a').write(log + '\n')

        if curr_epoch % 1 == 0 and not cmd_args.bestonly:
            checkpoint = {
                'model': net.module.state_dict() if len(gpu_ids.split(',')) > 1 else net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            
        current_mae = val(net, curr_epoch)

        print('1 epoch time: %f' % (time.time() - start_time))

        net.train() 
        if current_mae < best_mae:
            best_mae = current_mae
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_path, exp_name, 'best_mae.pth'))

        if curr_epoch > args['max_epoch']:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return
        curr_epoch += 1
        scheduler.step()  
        
def val(net, epoch):
    mae_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            exemplar, query, other = sample['exemplar'].cuda(), sample['query'].cuda(), sample['other'].cuda()
            exemplar_gt = sample['exemplar_gt'].cuda()
            
            output = preprocess.feature_pyramid_extract(sample['frames'].cuda())

            examplar_final, query_final, other_final, opflow_based_pred = net(exemplar, query, other, output['exempler_featmap'], output['query_featmap'], output['opflow_angle'], output['opflow_magnitude'])
            
            res = (examplar_final.data > 0).to(torch.float32).squeeze(0)
            
            mae = torch.mean(torch.abs(res - exemplar_gt.squeeze(0)))

            batch_size = exemplar.size(0)
            mae_record.update(mae.item(), batch_size)
            
        log = "val: iter: %d, mae: %f5" % (epoch, mae_record.avg)
        print(log)
        open(val_log_path, 'a').write(log + '\n')
        return mae_record.avg

if __name__ == '__main__':
    main()