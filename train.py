# Single action per state, use pretrained model, predict 100-bin
import os
import sys
import time
import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose
from torch.distributions import Normal

import utils
from dataset import ProstateDataset, ProstateDataset2
from scheduler import WarmupMultiStepLR
import transforms as T
import network 
import yaml
import gc

from timm.data.loader import MultiEpochsDataLoader

step = 0

def train_one_epoch(model, optimizer, lr_scheduler, criterion,
                    dataloader, device, epoch, print_freq, writer, scaler, 
                    mixed_precision=True):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data, label in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device) 

        with torch.autocast('cuda', enabled=mixed_precision):
            output = model(data)
            loss, loss_g1v, loss_g2v = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        loss_g1v_val = loss_g1v.item()
        loss_g2v_val = loss_g2v.item()
        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, loss_g1v=loss_g1v_val, 
            loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_g1v', loss_g1v_val, step)
            writer.add_scalar('loss_g2v', loss_g2v_val, step)
        step += 1
        lr_scheduler.step()


def evaluate(model, criterion, dataloader, device, writer, print_freq, mixed_precision=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, print_freq, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=mixed_precision):
                output = model(data)
                loss, loss_g1v, loss_g2v = criterion(output, label)
            metric_logger.update(loss=loss.item(), 
                loss_g1v=loss_g1v.item(), 
                loss_g2v=loss_g2v.item())

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_g1v', metric_logger.loss_g1v.global_avg, step)
        writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)
    return metric_logger.loss.global_avg


def main(args):
    global step

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    utils.mkdir(args.output_path) # create folder to store checkpoints
    utils.init_distributed_mode(args) # distributed mode initialization

    # Set up tensorboard summary writer
    train_writer, val_writer = None, None
    utils.mkdir(args.log_path) # create folder to store tensorboard logs
    if not args.distributed or (args.rank == 0 and args.local_rank == 0):
        train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
        val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                    
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')
    log_data_min = T.log_transform(ctx['data_min'], k=args.k).astype('float32')
    log_data_max = T.log_transform(ctx['data_max'], k=args.k).astype('float32')
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max)
    ])

    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    dataset_train = ProstateDataset(
        args.train_anno,
        transform_data=transform_data,
        transform_label=transform_label
    )

    print('Loading validation data')
    dataset_valid = ProstateDataset(
        args.val_anno,
        transform_data=transform_data,
        transform_label=transform_label
    )

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = SequentialSampler(dataset_train)
        valid_sampler = SequentialSampler(dataset_valid)

    # Use MultiEpochsDataLoader to reduce first batch long loading time per epoch
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, 
        persistent_workers=True,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size // 4,
        sampler=valid_sampler, num_workers=args.workers, 
        # prefetch_factor=1, persistent_workers=True, 
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")  

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    scaler = torch.GradScaler(enabled=args.mixed_precision)

    # Define loss function
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones=lr_milestones
        if checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, lr_scheduler, criterion, 
                        dataloader_train, device, epoch, args.print_freq, train_writer, 
                        scaler, args.mixed_precision)
        evaluate(model, criterion, dataloader_valid, device, val_writer, args.print_freq, args.mixed_precision)
        print('Before saving')
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'scaler': scaler.state_dict() if args.mixed_precision else None,
            'args': args
        }
        # Save checkpoint per epoch
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        print('After saving')
        # Save checkpoint every epoch block
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))
        gc.collect()
        torch.cuda.empty_cache()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Reinforcement Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='prostate', type=str, help='dataset name')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='../relevant_files', help='annotation files location')
    parser.add_argument('-t','--train-anno', default='prostate_train.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='prostate_val.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='../models', help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='../models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='prostate_inv', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-mc', '--model-config', type=str, help='inverse model config file (yaml)')

    # Training related
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.0004, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='number of warmup epochs')   
    parser.add_argument('-eb', '--epoch-block', type=int, default=10, help='epochs in a saved block')
    parser.add_argument('-nb', '--num-block', type=int, default=17, help='number of saved block')
    parser.add_argument('-j', '--workers', default=12, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-pf', '--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('-mp', '--mixed-precision', action='store_true', help='use mixed precision training')
    
    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)

    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block
    
    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
