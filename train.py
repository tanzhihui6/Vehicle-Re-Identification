import argparse      #命令行解释器
import os, sys       #操作
import shutil        #文件高级操作
import time          #时间模块
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn    #使用GPU

import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed     #分布式训练
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
from dataset.dataset import VeriDataset # AicDataset

#取出torchvision.models中所有集成好的模型名称，并按照升序排列
#取出model.__dict__中所有对象的名字
#名字全部小写并且不是以__开头
#判断对象是否可调用
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

'''
返回 ['alexnet', 'densenet121', 'densenet161', 'densenet169', 
'densenet201', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 
'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18',
 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0',
   'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 
'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']
'''

parser = argparse.ArgumentParser(description='PyTorch Relationship') #命令行解析的主要入口点

parser.add_argument('data', metavar='DIR', help='path to dataset') #数据集的路径
parser.add_argument('trainlist', metavar='DIR', help='path to test list') #trainlist路径
parser.add_argument('--dataset',  default='veri', type=str,
                    help='dataset name veri or aic (default: veri)')#选择数据集名称
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')#用几线程参与运算？没懂
#定义batchsize
parser.add_argument('--batch_size', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
#epoch的初始值
parser.add_argument('--start_epoch',  default=0, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
#定义骨干网络
parser.add_argument('--backbone',  default='resnet50', type=str,
                    help='backbone network resnet50 or resnet101 (default: resnet50)')
#模型权重的存储位置？
parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to weights (default: none)')
#？
parser.add_argument('--scale-size',default=256, type=int,
                    help='input size')
#待识别的种类数量
parser.add_argument('-n', '--num_classes', default=16, type=int, metavar='N',
                    help='number of classes / categories')
#是否保存输出
parser.add_argument('--write-out', dest='write_out', action='store_true',
                    help='write scores')
#这是调整之后的图片尺寸？
parser.add_argument('--crop_size',default=224, type=int,
                    help='crop size')
parser.add_argument('--val_step',default=1, type=int,
                    help='val step')
#模型训练的轮次
parser.add_argument('--epochs',default=200, type=int,
                    help='epochs')
#存储模型的路径
parser.add_argument('--save_dir',default='./checkpoints/att/', type=str,
                    help='save_dir')
#GPU数量？
parser.add_argument('--num_gpu', default=4, type=int, metavar='PATH',
                    help='path for saving result (default: none)')

best_prec1 = 0

"""函数作用应该是读取数据进行预处理和数据增强
输入：
     dataset_name 数据集名字
     data_dir 数据集路径
     train_list 训练集的标签数据（每张图片名对应的标签）
输出：
     data_loader
"""
def get_dataset(dataset_name, data_dir,train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    if dataset_name == 'veri':
        train_data_transform = transforms.Compose([
                transforms.Scale((336,336)),#改变图的尺寸为336×336
                transforms.RandomHorizontalFlip(),#随机水平翻转，一半的概率翻转，一半的概率不翻转
                transforms.RandomCrop((crop_size,crop_size)),#随机选取中心点的位置进行切割
                transforms.ToTensor(),#转换为张量，转换之后数据范围是0-1
                normalize])  #归一化
        #得到dataset数据迭代器
        train_set = VeriDataset(data_dir, train_list, train_data_transform, is_train= True )
    elif dataset_name == 'aic':
        train_data_transform = transforms.Compose([
                transforms.Scale((336,336)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size,crop_size)),
                transforms.ToTensor(),
                normalize])
        train_set = AicDataset(data_dir, train_list, train_data_transform, is_train= True )
    else:
        print("!!!dataset error!!!")
        return
    #生成数据迭代器
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=True)
    
    return train_loader
    
def main():
    
    global args, best_prec1
    args = parser.parse_args() #把命令行的参数转化为具有属性的对象
    print (args)
    best_acc = 0
    # Create dataloader
    print ('====> Creating dataloader...')
    
    data_dir = args.data
    train_list = args.trainlist
    dataset_name = args.dataset
    #获得data_loader
    train_loader = get_dataset(dataset_name, data_dir,  train_list)
    # load network 加载网络
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_classes)
    elif args.backbone == 'resnet101':    
        model = resnet101(num_classes=args.num_classes)

    # 如果有预训练的权重，则加载权重
    if args.weights != '':
        try:
            ckpt = torch.load(args.weights)
            model.module.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !! path is ',args.weights)
        except Exception as e:
            model_init(args.weights,model)
    model = torch.nn.DataParallel(model) #并行计算
    model.cuda()
    mkdir_if_missing(args.save_dir) #创建保存文件的目录
    #定义优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = 10e-3 )
    #定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    #GPU加速用的flag，能够优化运行效率？
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs + 1):
        #根据训练轮次调整学习率
        adjust_lr(optimizer, epoch)
        train (train_loader, model, criterion,optimizer, epoch)
        #训练一定轮次后保存模型去验证
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
    '''
    初始化模型，加载除全连接层以外的权重。
    '''
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
'''
定义调整学习率的函数
学习率调整策略：30为间隔，逐渐变小
'''
def adjust_lr(optimizer, ep):
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
        #param_groups是optimizer的参数组，以字典的形式管理参数
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
    #训练模式下，启用BN和Dropout
    model.train()


    for i, (image, target, camid) in enumerate(train_loader):
        
        batch_size = image.shape[0]

        target = target.cuda()
        image = torch.autograd.Variable(image, volatile=True).cuda()
        output, feat = model(image)
        
        loss = criterion(output, target)

        prec1= accuracy(output, target, topk=(1,5 ))
        losses.update(loss.item(), image.size(0)) #每一轮训练的平均损失
        top1.update(prec1[0], image.size(0))  #top1的平均准确度
        top5.update(prec1[1], image.size(0)) #top5的平均准确度


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
#保存相应epoch的模型
def save_checkpoint(model,epoch,optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath =  osp.join(args.save_dir, 'Car_epoch_' + str(epoch) + '.pth')
    torch.save(state, filepath)
#判断是否存在相应目录，不存在就创建一个
def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#返回一个batch里面的topk准确率
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

'''
定义AverageMeter对象，返回平均值avg
'''
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
