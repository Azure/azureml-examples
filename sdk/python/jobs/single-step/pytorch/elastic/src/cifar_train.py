import argparse
import io
import os
import math
import time
from enum import Enum
from PIL import Image

import mlflow

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.io import MultiImage
from azureml.core import Run
from tqdm import tqdm


print("Starting script")
print(f"World size: {os.environ['WORLD_SIZE']}")


NUM_EPOCHS = 40


current_run = Run.get_context()


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 = 0


def get_model_checkpoint_path(args):
    res = f"{args.checkpoint_dir}/checkpoint.pt"
    print(f"Model checkpoint path: {res}")
    return res


def get_model_checkpoint_epoch_path(args, epoch):
    res = f"{args.checkpoint_dir}/epoch{epoch}.pt"
    print(f"Model checkpoint epoch path: {res}")
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--dataset_directory_name", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=NUM_EPOCHS, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--output-dir', default='outputs', type=str,
                        help='path to store the model outputs') 
    args = parser.parse_args()

    return args



def make_dataset(root_folder, subset):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if subset == "train":
        train_set = torchvision.datasets.CIFAR10(root=root_folder, train=True, download=False, transform=transform)
        return train_set
    
    if subset == "val":
        test_set = torchvision.datasets.CIFAR10(root=root_folder, train=False, download=False, transform=transform)
        return test_set
        
    return None


class WarmUpCosineLRCalculator:

    def __init__(self, batches_per_epoch, lr, epochs=NUM_EPOCHS):
        self._warmup_steps: int = batches_per_epoch * 3
        self._total_steps: int = batches_per_epoch * epochs
        self._cycles: float = 0.45
        self._lr = lr
    
    def calculate(self, step):
        if step < self._warmup_steps:
            return float(step) / float(max(1.0, self._warmup_steps))
        # progress after warmup
        progress = float(step - self._warmup_steps) / float(max(1, self._total_steps - self._warmup_steps))
        lambda_ = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self._cycles) * 2.0 * progress)))
        return self._lr * lambda_


def main(args):
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = True if args.world_size > 1 else False
    main_worker(args)


def main_worker(args):
    global best_acc1
    args.gpu = int(os.environ['LOCAL_RANK'])
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size)
    args.workers = int(os.cpu_count() / torch.cuda.device_count()) * 3
    print(f'Num workers: {args.workers}')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    device = torch.device('cuda:{}'.format(args.gpu))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr * args.world_size,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    model_checkpoint_path = get_model_checkpoint_path(args)
    if os.path.isfile(model_checkpoint_path):
            print("=> loading checkpoint '{}'".format(model_checkpoint_path))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(model_checkpoint_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(model_checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_checkpoint_path))
        
    print(f"=> Loading data from {args.dataset_directory_name}")
    train_dataset = make_dataset(args.dataset_directory_name, "train")
    val_dataset = make_dataset(args.dataset_directory_name, "val")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader_dist = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    lr_calculator = WarmUpCosineLRCalculator(len(train_loader), args.lr * args.world_size, args.epochs)

    if os.environ["RANK"] == 0:
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("momentum", args.momentum)
    
    if args.evaluate:
        validate(val_loader_dist, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print(f'Starting epoch {epoch}')
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, device, lr_calculator)
        acc1 = validate(val_loader_dist, model, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.rank == 0:
            save_checkpoint({
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, epoch, args)
            
    if os.environ["RANK"] == 0:
        mlflow.pytorch.log_model(model, "model")
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model, model_path)
        print(f"Succesfully trained model. Model saved to {args.output_dir}/model.pt")


def train(train_loader, model, criterion, optimizer, epoch, device, lr_calculator):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    num_batches = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        lr = lr_calculator.calculate(i + epoch * num_batches)
        optimizer.param_groups[0]['lr'] = lr
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        if os.environ["RANK"] == 0:
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(tqdm(loader)):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    print(f'Top1 avg: {top1.avg}')
    print(f'Top5 avg: {top5.avg}')

    if os.environ['RANK'] == '0':
        mlflow.log_metric('world_size', args.world_size)
        mlflow.log_metric('top1', top1.avg)
        mlflow.log_metric('top5', top5.avg)

    return top1.avg


def save_checkpoint(state, epoch, args): 
    model_checkpoint_path = get_model_checkpoint_path(args)
    model_checkpoint_epoch_path = get_model_checkpoint_epoch_path(args, epoch)
    state['epoch'] = epoch + 1
    start = time.time()
    torch.save(state, model_checkpoint_path)
    print(f"Saved checkpoint to {model_checkpoint_path}")
    print(f'Checkpointing time: {time.time() - start}')
    start = time.time()
    torch.save(state, model_checkpoint_epoch_path)
    print(f"Saved epoch checkpoint to {model_checkpoint_epoch_path}")
    print(f'Epoch Checkpointing time: {time.time() - start}')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Finished training...")
