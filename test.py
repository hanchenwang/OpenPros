import os
import sys
import time
import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose
import numpy as np

import utils
import network
from dataset import ProstateDataset
import transforms as T
import pytorch_ssim
import yaml
from tqdm import tqdm, trange
from torchmetrics.functional import pearson_corrcoef
from vis import plot_sos
# from scipy.stats import pearsonr


def evaluate(model, dataloader, device, k, ctx,
                vis_path, vis_batch, vis_sample, missing, std):
    model.eval()
    
    label_tensor, label_pred_tensor = [], [] # store normalized prediction & gt in tensor
    if missing or std:
        data_list, data_noise_list = [], [] # store original data and noisy/muted data

    with torch.no_grad():
        batch_idx = 0
        for data, label in tqdm(dataloader, desc='Testing', total=len(dataloader)):
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)
            
            if missing or std:
                # Add gaussian noise
                data_noise = torch.clip(data + (std ** 0.5) * torch.randn(data.shape).to(device, non_blocking=True), min=-1, max=1)

                # Mute some traces
                mute_idx = np.random.choice(data.shape[3], size=missing, replace=False) 
                data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]
                
                data_np = T.tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=k)
                data_noise_np = T.tonumpy_denormalize(data_noise,  ctx['data_min'], ctx['data_max'], k=k)
                data_list.append(data_np)
                data_noise_list.append(data_noise_np)
                pred = model(data_noise)
            else:
                pred = model(data)

            label_pred_tensor.append(pred)

            # Visualization
            if vis_path and batch_idx < vis_batch:
                label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
                label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
                np.save(f'{vis_path}/pred_{batch_idx}.npy', label_pred_np)
                for i in range(vis_sample):
                    plot_sos(label_np[i, 0], label_pred_np[i, 0], f'{vis_path}/V_{batch_idx}_{i}.png', exp=True) #, vmin=ctx['label_min'], vmax=ctx['label_max'])
            batch_idx += 1
        label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
        del label_tensor, label_pred_tensor
        torch.cuda.empty_cache()
        l1 = nn.L1Loss()
        l2 = nn.MSELoss()
        print(f'MAE: {l1(label_t, pred_t)}')
        print(f'MSE: {l2(label_t, pred_t)}')
        print(f'RMSE: {torch.sqrt(l2(label_t, pred_t))}')
        pcc_list = []
        for i in trange(label_t.shape[0], desc='Computing PCC'):
            pcc = pearson_corrcoef(label_t[i].flatten(), pred_t[i].flatten())
            pcc_list.append(pcc)
        print(f'PCC: {torch.mean(torch.stack(pcc_list))}')
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        ssim = 0
        chunk_size = 128
        label_t = label_t / 2 + 0.5
        pred_t = pred_t / 2 + 0.5
        for i in trange(label_t.shape[0] // chunk_size + 1, desc='Computing SSIM'): 
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, label_t.shape[0])
            num_samples = end_idx - start_idx
            ssim += num_samples * ssim_loss(label_t[start_idx:end_idx], pred_t[start_idx:end_idx])
        ssim /= label_t.shape[0]        
        print(f'SSIM: {ssim}')


def main(args):

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    print("Loading data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k).astype('float32')
    log_data_max = T.log_transform(ctx['data_max'], k=args.k).astype('float32')
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    print('Loading validation data')
    dataset_valid = ProstateDataset(
        args.val_anno,
        root=args.dataset_root,
        transform_data=transform_data,
        transform_label=transform_label
    )

    print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
     
    try:
        with open(args.model_config) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('Model config file not found.')
        sys.exit()
    except yaml.YAMLError:
        print('Error in model config file.')
        sys.exit()

    model = network.model_dict[args.model](**model_config).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
        print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))
    
    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None
    
    print("Start testing")
    start_time = time.time()
    evaluate(model, dataloader_valid, device, args.k, ctx, 
                vis_path, args.vis_batch, args.vis_sample, args.missing, args.std)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='prostate', type=str, help='dataset name')
    parser.add_argument('-dr', '--dataset-root', default='/work/nvme/beej/ylin20/jhu_data/all_data', type=str, help='dataset name')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='../relevant_files', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='prostate_test.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='../models', help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='prostate_inv', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-mc', '--model-config', type=str, help='inverse model config file (yaml)')

    # Test related
    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1e9, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_true")
    parser.add_argument('-vsu','--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb','--vis-batch', help='number of batch to be visualized', default=0, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=0, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
    torch.cuda.empty_cache()
