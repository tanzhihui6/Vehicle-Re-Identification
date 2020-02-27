import argparse
import os, sys
import shutil
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import os.path as osp
from torch.autograd import Variable
import math
from networks.resnet import resnet50, resnet101
from dataset.dataset import VeriDataset
#from dataset.dataset import AicDataset

"""
此文件调用dataset中创建的对象读取数据创建dataloader，然后调用networks中创建的模型对象来创建对应的resnet模型。
对模型进行一些初始化设置后，根据用户提供的输入信息，对模型进行训练，并把模型训练的结果(model weights)保存在指定文件夹中。
默认保存路径为：当前目录下checkpoints/att文件夹。

这里用户需要提供2个必要参数：
- data: 图片文件存储的路径
- trainlist: txt文件的路径，里面存储training数据中包含的图片的id

剩下的为选择性参数，如果用户不提供，则采用默认值：
- dataset: 数据集的类型，默认为Veri (Vehicle Re-identification) 数据集
- workers: 指定workers的数量来处理数据
- batch_size: 批处理的大小
- start_epoch: epoch的初始值
- backbone: 采用什么模型， 默认为resnet50
- weights： 模型的权重（如果有，则提供该权重的存储路径）
- scale-size： 模型scale后的大小
- num_classes: 有多少种labels
- write-out： 是否保存输出
- crop_size: 模型裁剪后的大小
- epochs: 模型要训练的次数
- save_dir: 模型训练后权重保存的路径
- num_gpu: 规定用于训练模型的gpu的数量
"""

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('trainlist', metavar='DIR', help='path to test list')
parser.add_argument('--dataset',  default='veri', type=str,
                    help='dataset name veri or aic (default: veri)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--backbone',  default='resnet50', type=str,
                    help='backbone network resnet50 or resnet101 (default: resnet50)')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to weights (default: none)')
parser.add_argument('--scale-size',default=256, type=int,
                    help='input size')
parser.add_argument('-n', '--num_classes', default=16, type=int, metavar='N',
                    help='number of classes / categories')
parser.add_argument('--write-out', dest='write_out', action='store_true',
                    help='write scores')
parser.add_argument('--crop_size',default=224, type=int,
                    help='crop size')
parser.add_argument('--val_step',default=1, type=int,
                    help='val step')
parser.add_argument('--epochs',default=200, type=int,
                    help='epochs')
parser.add_argument('--save_dir',default='./checkpoints/att/', type=str,
                    help='save_dir')
parser.add_argument('--num_gpu', default=4, type=int, metavar='PATH',
                    help='path for saving result (default: none)')

best_prec1 = 0

def get_dataset(dataset_name,data_dir,train_list):
    """
    用于初始化一个dataloader对象，来读取dataset中的指定数据。
    在这个函数中，对原图片根据imagenet图片的statistics进行了正则化处理。并且对图片进行了一些预处理
    :arg
        dataset_name: 数据集的类型
        data_dir: 图片数据存储的路径
        train_list:: txt文件的路径，里面存储training数据中包含的图片的id
    :returns
        data_loader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    if dataset_name == 'veri':
        train_data_transform = transforms.Compose([
                transforms.Scale((336,336)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size,crop_size)),
                transforms.ToTensor(),
                normalize])
        train_set = VeriDataset(data_dir, train_list, train_data_transform, is_train= True )
    # elif dataset_name == 'aic':
    #     train_data_transform = transforms.Compose([
    #             transforms.Scale((336,336)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomCrop((crop_size,crop_size)),
    #             transforms.ToTensor(),
    #             normalize])
    #     train_set = AicDataset(data_dir, train_list, train_data_transform, is_train= True )
    else:
        print("!!!dataset error!!!")
        return
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    
    return train_loader
    
def main():
    
    global args, best_prec1
    args = parser.parse_args()
    print (args)
    best_acc = 0
    # Create dataloader
    print ('====> Creating dataloader...')
    
    data_dir = args.data
    train_list = args.trainlist
    dataset_name = args.dataset

    train_loader = get_dataset(dataset_name, data_dir,  train_list)
    # load network
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_classes)
    elif args.backbone == 'resnet101':    
        model = resnet101(num_classes=args.num_classes)
    
    if args.weights != '':
        try:
            ckpt = torch.load(args.weights)
            model.module.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !! path is ',args.weights)
        except Exception as e:
            model_init(args.weights,model)
    model = torch.nn.DataParallel(model)
    model.cuda()
    mkdir_if_missing(args.save_dir)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 10e-3 )
    criterion = nn.CrossEntropyLoss().cuda()
    
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs + 1):
        # try not adjust learning rate
        #adjust_lr(optimizer, epoch)
        train (train_loader, model, criterion,optimizer, epoch)
        
        if epoch% args.val_step == 0:
            save_checkpoint(model,epoch,optimizer)
        '''
        if epoch% args.val_step == 0:
            acc = validate(test_loader, model, criterion)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, is_best=is_best,train_batch=60000, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
        '''

    return

def model_init(weights,model):
    '''
    print ('attention!!!!!!! load model fail and go on init!!!')
    ckpt = torch.load(weights)
    pretrained_dict=ckpt['state_dict']
    model_dict = model.module.state_dict()
    model_pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(model_pre_dict)
    model.module.load_state_dict(model_dict)
    for v ,val in model_pre_dict.items() :
        print ('update',v)
    '''
    """
    加载模型保存的权重
    :arg
        weights: 模型权重的路径
        model: 模型对象
    """
    saved_state_dict = torch.load(weights)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        # print(i_parts)
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        else:
            print ('Not Load',i)
    model.load_state_dict(new_params)

    print ('-------Load Weight',weights)

def adjust_lr(optimizer, ep):
    """
    用于随着训练次数的增加来减少学习率
    :arg
        optimizer: 用于训练模型的优化器对象
        ep: epochs
    """
    if ep < 10:
        lr = 1e-4 * (ep + 1) / 2
    elif ep < 40:
        lr = 1e-3 
    elif ep < 70:
        lr = 1e-4 
    elif ep < 100:
        lr = 1e-5 
    elif ep < 130:
        lr = 1e-6
    elif ep < 160:
        lr = 1e-4 
    else:
        lr = 1e-5 
    for p in optimizer.param_groups:
        p['lr'] = lr


    print ("lr is ",lr)

def train(train_loader, model, criterion,optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    # switch to train mode
    model.train()


    for i, (image, target, camid) in enumerate(train_loader):
        
        batch_size = image.shape[0]

        target = target.cuda()
        image = torch.autograd.Variable(image, volatile=True).cuda()
        output, feat = model(image)
        
        loss = criterion(output, target)

        prec1= accuracy(output, target, topk=(1,5 ))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1[0], image.size(0))
        top5.update(prec1[1], image.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 1==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
               
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    lated = 0
    val_label = []
    val_pre = []
    
    model.eval()

    end = time.time()
    tp = {} # precision
    p = {}  # prediction
    r = {}  # recall
    
    for i, (union, obj1, obj2, bpos, target, full_im, bboxes_14, categories) in enumerate(val_loader):
        
        batch_size = bboxes_14.shape[0]
        cur_rois_sum = categories[0,0]
        bboxes = bboxes_14[0, 0:categories[0,0], :]
        for b in range(1, batch_size):
            bboxes = torch.cat((bboxes, bboxes_14[b, 0:categories[b,0], :]), 0)
            cur_rois_sum += categories[b,0]
        assert(bboxes.size(0) == cur_rois_sum), 'Bboxes num must equal to categories num'

        target = target.cuda()
        union_var = torch.autograd.Variable(union, volatile=True).cuda()
        obj1_var = torch.autograd.Variable(obj1, volatile=True).cuda()
        obj2_var = torch.autograd.Variable(obj2, volatile=True).cuda()
        bpos_var = torch.autograd.Variable(bpos, volatile=True).cuda()
        full_im_var = torch.autograd.Variable(full_im, volatile=True).cuda()
        bboxes_var = torch.autograd.Variable(bboxes, volatile=True).cuda()
        categories_var = torch.autograd.Variable(categories, volatile=True).cuda()
        
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, bboxes_var, categories_var)
        
        

        # compute output
        loss = criterion(output, target)

        prec1 = accuracy(output, target, topk=(1,))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        val_label[lated:lated + batch_size] =target
        val_pre [lated:lated+batch_size] = pred.data.cpu().numpy().tolist()[:]
        lated = lated + batch_size
        
        losses.update(loss.item(), obj1.size(0))
        top1.update(prec1[0], obj1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 8==0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))


    print ('----------------------------------------------------------')
    count = [0]*16
    acc = [0]*16
    pre_new = []
    for i in val_pre:
        for j in i:
            pre_new.append(j)
    for idx in range(len(val_label)):
        count[val_label[idx]]+=1
        if val_label[idx] == pre_new[idx]:
            acc[val_label[idx]]+=1
    classaccuracys = []
    for i in range(16):
        if count[i]!=0:
            classaccuracy = (acc[i]*1.0/count[i])*100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)

    print(('Testing Results: Prec@1 {top1.avg:.3f} classacc {classaccuracys} Loss {loss.avg:.5f}'
          .format(top1=top1, classaccuracys = classaccuracys, loss=losses)))

    return top1.avg[0]

def save_checkpoint(model,epoch,optimizer):
    """
    保存训练后的模型
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath =  osp.join(args.save_dir, 'Car_epoch_' + str(epoch) + '.pth')
    torch.save(state, filepath)

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    def __init__(self):
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


if __name__=='__main__':
    main()
