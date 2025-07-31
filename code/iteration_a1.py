from collections import OrderedDict

import torch
from torch import nn

# !/user/bin/python
# coding=utf-8

import os, sys
from statistics import mode



import numpy as np
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import importlib
def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
device = get_default_device()

if device == 'mps':
    train_root = "/Users/mac/Downloads/6.UAED"
else:
    train_root = "/home/xjx/6.UAED"
sys.path.append(train_root)
print("Now running at:", device)

matplotlib.use('Agg')   #设置后端为Agg

from data.data_loader_one_random_uncert1 import BSDS_RCFLoader

MODEL_NAME = "models.sigma_logit_unetpp"

Model = importlib.import_module(MODEL_NAME)

from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
import random
import numpy
from torch.autograd import Variable
import ssl
from utils_yyz import *

ssl._create_default_https_context = ssl._create_unverified_context
from torch.distributions import Normal, Independent

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=8, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--LR', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,  #学习率衰减步长
                    metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=10, type=int, metavar='N',        #最大训练轮次
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',      #起始轮次
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=50, type=int,       #打印频率
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID(s), e.g., "0,1" for multi-GPU')
parser.add_argument('--tmp', help='tmp folder', default='result/bsds/round2/a1')  #临时文件夹路径
parser.add_argument('--dataset', help='root folder of dataset', default='dataset/bsds')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--std_weight', default=1, type=float, help='weight for std loss')  #标准差损失的权重

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')  #输出分布类型 高斯分布

parser.add_argument('--resume', default=True, type=bool, help='pretrain')
parser.add_argument('--path', default="result/bsds/round2/a1/epoch-19-training-record/epoch-16-checkpoint.pth", type=str)


args = parser.parse_args() #解析参数并存到args里

if device == 'cuda':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 根据 PCI 总线 ID 选择设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   # 设置可见 GPU（可以选择多个，像 '0,1'）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取当前脚本的绝对路径
THIS_DIR = abspath(dirname(__file__))

# 设置 TMP_DIR 为临时文件夹
TMP_DIR = join(THIS_DIR, args.tmp)

# 如果目录不存在，创建它
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
#复制环境
# file_name = os.path.basename(__file__)
# copyfile(join(train_root, MODEL_NAME[:6], MODEL_NAME[7:] + ".py"), join(TMP_DIR, MODEL_NAME[7:] + ".py"))
# copyfile(join(train_root, file_name), join(TMP_DIR, file_name))


random_seed = 555 #设置了随机种子
if random_seed > 0:
    random.seed(random_seed)  #设置Python内置的random模块的随机数生成器的种子
    torch.manual_seed(random_seed) #设置PyTorch的CPU随机数生成器的种子
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed) #设置NumPy的随机数生成器的种子。

def adjust_weights_based_on_mask(weights, loss_mask):
    # 根据 loss_mask 调整权重
    weights = weights.clone()  # 避免修改原始权重
    weights[loss_mask == 5 ] *= 10# 增大权重
    weights[loss_mask == 4] *= 5  # 增大权重
    weights[loss_mask == 3] *= 2  # 减小权重
    weights[loss_mask == 2] *= 1 # 减小权重
    weights[loss_mask == 1] *= 0  # 减小权重
    # 1和4的权重保持不变
    return weights

def adjust_weights_based_on_epmask(weights, ep_mask):
    # 根据 loss_mask 调整权重
    weights = weights.clone()  # 避免修改原始权重
    weights[ep_mask == 1] *= 5# 增大权重
    weights[ep_mask == 0.498] *= 1  # 增大权重

    # 1和4的权重保持不变
    return weights

def cross_entropy_loss_RCF_update(prediction, labelef, std, ada, loss_mask):  # 二元交叉熵损失
    label = labelef.long()  # labelef转换为label
    mask = label.float()  # 创建mask，通过将label转换为浮点值。

    # 定义忽略类的值和容差
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    num_two = torch.sum((mask == 2).float()).float()

    # 确保所有标签的总数等于label张量的元素总数
    assert num_negative + num_positive + num_two == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    # 确认没有标签值为2的元素
    assert num_two == 0
    # 根据正类和负类的数目调整了mask张量中对应位置的权重
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    new_mask = mask * torch.exp(std * ada)
    adjusted_weights = adjust_weights_based_on_mask(new_mask, loss_mask)
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=adjusted_weights.detach(), reduction='sum')

    return cost, mask




def cross_entropy_loss_RCF(prediction, labelef, std, ada):  # 二元交叉熵损失
    label = labelef.long()  # labelef转换为label
    mask = label.float()  # 创建mask，通过将label转换为浮点值。

    # 定义忽略类的值和容差
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    num_two = torch.sum((mask == 2).float()).float()

    # 确保所有标签的总数等于label张量的元素总数
    assert num_negative + num_positive + num_two == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
    # 确认没有标签值为2的元素
    assert num_two == 0
    # 根据正类和负类的数目调整了mask张量中对应位置的权重
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    new_mask = mask * torch.exp(std * ada)
   # adjusted_weights = adjust_weights_based_on_mask(new_mask, loss_mask, mask)
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum')

    return cost, mask


def step_lr_scheduler(optimizer, epoch, init_lr=args.LR, lr_decay_epoch=3):  #优化器 、 训练轮次 、 初始学习率 、 学习率衰减周期
    """Decay learning rate by a factor of 0.9_mask_a1+a2.9_mask_a1.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def reverse_dict(mask, train_dataset): # 已经创建数字索引了
    # 创建字典进行0-1599和.png双向映射
    # name: images/train1to4-bai/100075_1.1to4png groundTruth/edge/1th100075_1.1to4png groundTruth/edge/2th100075_1.1to4png
    file_dict = {name.split()[0].split('/')[-1]: idx for idx, name in enumerate(train_dataset.filelist)}
    reverse_file_dict = {idx: name.split()[0].split('/')[-1] for idx, name in enumerate(train_dataset.filelist)}
    # 双向映射字典
    bi_dict = {**file_dict, **reverse_file_dict}
    return bi_dict

def save_mask_as_png(mask_binary, output_dir, bi_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个mask
    for idx in range(mask_binary.shape[0]):
        mask = mask_binary[idx].cpu().numpy() * 255  # 将mask转为numpy数组并缩放到[0.9_mask_a1+a2.9_mask_a1, 255]
        # print(f"Saving mask_a1+a2 {idx} with shape {mask_a1+a2.shape}")
        mask_image = Image.fromarray(mask.astype(np.uint8), mode='L')  # 转换为PIL图像
        img_name = bi_dict[idx]
        mask_image.save(os.path.join(output_dir, f"mask_{img_name}"))  # 保存为PNG图像

def main():

    if device == 'cuda':
        args.cuda = True

    #读取测试集和训练集
    train_dataset = BSDS_RCFLoader(root=args.dataset, split="train")
    # test_dataset = BSDS_RCFLoader(root=args.dataset, split="test")


    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=0, drop_last=True, shuffle=False)


    # test_loader = DataLoader(
    #     test_dataset, batch_size=1,
    #     num_workers=0.9_mask_a1+a2.9_mask_a1, drop_last=True, shuffle=False)
    # with open('../../../dataset/bsds/test.lst', 'r') as f:
    #     test_list = f.readlines()
    # test_list = [split(i.rstrip())[1] for i in test_list]
    print(train_dataset)
    # assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))
    mask = torch.zeros([len(train_loader) * args.batch_size, 160, 256], device=device)  # 设置掩膜用来赋值
    running_predictions = torch.zeros_like(mask, device=device)
    labels = torch.zeros_like(mask, device=device)
    cv_mask1 = torch.zeros_like(mask, device=device)


    bi_dict = reverse_dict(mask, train_dataset)
    # model
    model = Model.Mymodel(args).to(device)
    # 加载VoC上的预训练模型
    if args.resume:
        checkpoint = torch.load(args.path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    #更新日志
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('Adam', args.LR)))
    sys.stdout = log
    # save_iter_dir = os.path.join(TMP_DIR, 'iter')
    save_iter_dir = TMP_DIR

    if not os.path.exists(save_iter_dir):
        os.makedirs(save_iter_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    iter_begin_epoch = 0
    for epoch in range(args.start_epoch, args.maxepoch):
        # if epoch==0.9_mask_a1+a2.9_mask_a1:
        #     test(model, test_loader, epoch=epoch, test_list=test_list,
        #     save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        train(train_loader, model, optimizer, epoch,  # 在一个 epoch 内使用训练数据集对模型进行训练
              save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch), running_predictions=running_predictions,
              bi_dict=bi_dict, iter_begin_epoch=iter_begin_epoch, labels=labels, cv_mask1=cv_mask1)
        #test(model, test_loader, epoch=epoch, test_list=test_list,  # epoch 结束后，代码会调用 test 函数对模型进行评估，使用的是测试数据集。测试结果同样会被保存到指定的目录。
             #save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        # multiscale_test(model, test_loader, epoch=epoch, test_list=test_list, #进行多尺度测试
        #                 save_dir=join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush()  # write log
    # iter(running_predictions, args.maxepoch - iter_begin_epoch, save_dir=save_iter_dir, bi_dict=bi_dict, threshold=0.8,cv_mask=cv_mask)




def save_tensor_to_txt(tensor, file_path):
    """将tensor保存为txt文件"""
    tensor_cpu = tensor.detach().cpu()
    np.savetxt(file_path, tensor_cpu.numpy(), fmt='%d')

def train(train_loader, model, optimizer, epoch, save_dir, running_predictions, bi_dict, iter_begin_epoch,labels,cv_mask1):
    global cv_mask
    cv_mask = torch.zeros([0 * args.batch_size, 160, 256], device=device)
    optimizer = step_lr_scheduler(optimizer, epoch)     #调整学习率
    #初始化平均值计算器
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # 切换到模型的训练模式
    model.train()
    #打印当前的轮数和学习率
    print(epoch, optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time() #记录开始时间
    epoch_loss = [] #存储损失值
    counter = 0 #用于存储处理了多少batch
    ###########
    # ep_mask_images = get_all_ep_mask_image_paths(r'E:\UAED\6.UAED\6.UAED\result\bsds\round2\a1+a2\ mask_a1+a2')  # 自定义函数返回所有 ep_mask 图片路径
    # num_ep_images = len(ep_mask_images)  # 总图片数
    images_per_batch = 8  # 每个 batch 需要 8 张图片
###################
    #这里遍历 train_loader，获取每个 batch 的数据，包括图像 image、真实标签 label、标签的均值 label_mean 和标准差 label_std。
    # i 从 0.9_mask_a1+a2.9_mask_a1-100，里面才是一个batch
    for i, (image, label, label_mean, label_std, original_size, names) in enumerate(train_loader): # 有batch_size个
        names = list(names)
        # 计算当前 batch 的图片索引范围
        #################
        # start_idx = i * images_per_batch
        # end_idx = start_idx + images_per_batch
        #
        # # 防止索引超出总图片数
        # if start_idx >= num_ep_images:
        #     break
        # if end_idx > num_ep_images:
        #     end_idx = num_ep_images

        # 获取当前 batch 的 ep_mask 图片路径
        # current_ep_mask_paths = ep_mask_images[start_idx:end_idx]

        # 加载并拼接 ep_mask
        # ep_mask_list = []
        # for img_path in current_ep_mask_paths:
        #     img = Image.open(img_path).convert('L')  # 转为灰度图片
        #     img_tensor = torchvision.transforms.ToTensor()(img)  # 转为张量
        #     ep_mask_list.append(img_tensor)

        # 拼接为 [8, 1, 160, 256]
        # ep_mask = torch.stack(ep_mask_list).to(device)


        # 如果不足 8 张图片，补零张量
        # if len(ep_mask_list) < images_per_batch:
        #     padding = torch.zeros((images_per_batch - len(ep_mask_list), 1, 160, 256), device=device)
        #     ep_mask = torch.cat((ep_mask, padding), dim=0)
#################################
        # measure data loading time
        #更新数据加载时间
        data_time.update(time.time() - end)
        image, label, label_std = image.to(device), label.to(device), label_std.to(device)

        mean, std = model(image)
        #output = mean
        #根据预测的均值和标准差构建一个正态分布 Normal，然后使用 Independent 包装器增加分布的独立维度，这通常用于处理多维输出。
        outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
        #从分布中随机采样，并应用 Sigmoid 函数将采样结果转换为概率值。
        outputs = torch.sigmoid(outputs_dist.rsample())
        loss_mask = get_mask(outputs, names)
        loss_mask = torch.from_numpy(loss_mask)
        loss_mask_cv = loss_mask.squeeze(1)
        loss_mask_cv = loss_mask_cv.to(device)
        if epoch == args.maxepoch-1:
            cv_mask = torch.cat((cv_mask, loss_mask_cv), dim=0)

        #更新计数器：
        counter += 1
        #计算一个调整因子 ada，它随着 epoch 的增加而逐渐接近 1
        ada = (epoch + 1) / args.maxepoch

        if epoch % 2 == 0 :
            bce_loss, mask1 = cross_entropy_loss_RCF_update(outputs, label, std, ada,loss_mask)
        else:
            bce_loss, mask1 = cross_entropy_loss_RCF(outputs, label, std, ada)

        # bce_loss, mask1 = cross_entropy_loss_RCF(outputs, label, std, ada)
        # consist_loss,_ = cross_entropy_loss_RCF_update(outputs, label, std, ada,loss_mask)
        #bce_loss, mask1 = cross_entropy_loss_RCF(outputs, label, std, ada)
        std_loss = torch.sum((std - label_std) ** 2 * mask1)
        #预测标准差与标签标准差之间的平方差，乘以掩码 mask1
        loss = (bce_loss  + std_loss * args.std_weight) / args.itersize
        #反向传播
        loss.backward()
        #到达itersize的时候，进行模型参数更新
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        losses.update(loss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        # if epoch == args.maxepoch - 1: # 改成只写最后一个epoch
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            print(bce_loss.item(), std_loss.item())

            _, _, H, W = outputs.shape
            if epoch == args.maxepoch - 1:
                torchvision.utils.save_image(1 - outputs, join(save_dir, "iter-%d.jpg" % i))
                torchvision.utils.save_image(1 - mean, join(save_dir, "iter-%d_mean.jpg" % i))
                torchvision.utils.save_image(1 - std, join(save_dir, "iter-%d_std.jpg" % i))

        loss_mask = loss_mask.to(device)
        if epoch >= iter_begin_epoch:
            # 一次加载batch_size 16个
            for j in range(args.batch_size):
                name = names[j].split('/')[-1]  # 获取当前name,因为batch只有一个所以直接取0
                idx = bi_dict[name]

                # output.shape torch.Size([16, 1, 160, 256])
                _, _channel, _weight, _height = outputs.shape

                cv_mask1[idx, :label.shape[2], :label.shape[3]] = loss_mask[j, 0, :, :]
                labels[idx, :label.shape[2], :label.shape[3]] = label[j, 0, :, :]
                # 归一化后的标签值为127/255的像素点进行累积操作
                running_predictions[idx, :_weight, :_height] += outputs[j, 0, :, :]

    if epoch == args.maxepoch-1:
        save_iter_dir =r'E:\UAED\6.UAED\6.UAED\result\bsds\round2\a1'
        iter(running_predictions, args.maxepoch - 0, save_dir=save_iter_dir, bi_dict=bi_dict, threshold=0.8,
         cv_mask=cv_mask1,loss_mask=cv_mask1,label=labels)



    # # 暂时先不保存pth
    # if epoch == args.maxepoch - 1:
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

def iter(mask, len_epoch, save_dir, bi_dict, cv_mask,loss_mask,label,threshold=0.8):
    # 将 mask_a1+a2 除以 len_epoch

    # 确保保存目录存在
    device = mask.device  # 获取mask所在的设备
    loss_mask = loss_mask.to(device)
    label = label.to(device)

    normalized_mask = mask / len_epoch

    # 创建一个新的 mask_a1+a2，初始值为 normalized_mask
    mask_binary = normalized_mask.clone()

    # 打印归一化后的 mask_a1+a2
    print(f"Normalized mask_a1+a2: {normalized_mask}")

    # 创建一个新的 mask_a1+a2，初始值为0.5
    mask_binary = torch.full_like(normalized_mask, 0.5)
    print("mask_binary:", mask_binary.shape)
    print("normalized_mask:", normalized_mask.shape)
    print("loss_mask:", loss_mask.shape)
    print("label:", label.shape)
    # 将大于等于 threshold 的部分变成 0.9_mask_a1+a2.9_mask_a1.9, 代表是a2的
    #mask_binary[normalized_mask >= threshold] = 0.9
    # 大于0.8的变成 1  本来是上面这个
    print(label.shape, mask_binary.shape, normalized_mask.shape)
    mask_binary[(normalized_mask >= threshold) & (loss_mask >=3) & (label==1) ] = 1

    # struct = np.array([[1, 1, 1],
    #                    [1, 0, 1],
    #                    [1, 1, 1]], dtype=bool)
    # struct_3d = struct.reshape((1, 3, 3))
    # # 检查周围8邻域是否存在至少一个
    # label_cpu = label.cpu()
    # surrounding_has_one = binary_dilation(label_cpu, structure=struct_3d)
    # surrounding_has_one = torch.from_numpy(surrounding_has_one).to(device)
    # mask_binary[(normalized_mask >= threshold) & (loss_mask >= 4) & surrounding_has_one] = 1
    # 将小于 lower_threshold 的部分变成 0.9_mask_a1+a2.9_mask_a1.1,代表是a2的
    # mask_binary[normalized_mask < (1.0 - threshold)] = 0.1  原来是这个
    mask_binary[(normalized_mask < (1.0 - threshold)) &(loss_mask >=4) &(label==0) ] = 0
    print(f"Binary mask_a1+a2: {mask_binary}")
    # 保存处理后的 mask_a1+a2
    save_mask_dir = join(save_dir, 'mask_a11')
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)
    # 保存处理后的 mask_a1+a2
    save_mask_as_png(mask_binary, save_mask_dir, bi_dict)

def check_range(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return min_val.item(), max_val.item()

def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.to(device)

        # print(image.shape)
        mean, std = model(image)
        outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
        outputs = torch.sigmoid(outputs_dist.rsample())
        png = torch.squeeze(outputs.detach()).cpu().numpy()
        _, _, H, W = image.shape
        result = np.zeros((H + 1, W + 1))
        result[1:, 1:] = png
        filename = splitext(test_list[idx])[0]
        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir = os.path.join(save_dir, "1to4png")
        mat_save_dir = os.path.join(save_dir, "mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.1to4png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)

        mean = torch.squeeze(mean.detach()).cpu().numpy()
        result_mean = np.zeros((H + 1, W + 1))
        result_mean[1:, 1:] = mean
        result_mean_png = Image.fromarray((result_mean).astype(np.uint8))
        mean_save_dir = os.path.join(save_dir, "mean")

        if not os.path.exists(mean_save_dir):
            os.makedirs(mean_save_dir)
        result_mean_png.save(join(mean_save_dir, "%s.1to4png" % filename))

        std = torch.squeeze(std.detach()).cpu().numpy()
        result_std = np.zeros((H + 1, W + 1))
        result_std[1:, 1:] = std
        result_std_png = Image.fromarray((result_std * 255).astype(np.uint8))
        std_save_dir = os.path.join(save_dir, "std")

        if not os.path.exists(std_save_dir):
            os.makedirs(std_save_dir)
        result_std_png.save(join(std_save_dir, "%s.1to4png" % filename))


def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.6, 1, 1.6]
    for idx, image in enumerate(test_loader):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            # print('mulshape')
            # print(torch.unsqueeze(torch.from_numpy(im_).to(device), 0.9_mask_a1+a2.9_mask_a1).shape)
            mean, std = model(torch.unsqueeze(torch.from_numpy(im_).to(device), 0))
            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = torch.sigmoid(outputs_dist.rsample())
            result = torch.squeeze(outputs.detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result = np.zeros((H + 1, W + 1))
        result[1:, 1:] = multi_fuse
        filename = splitext(test_list[idx])[0]

        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir = os.path.join(save_dir, "1to4png")
        mat_save_dir = os.path.join(save_dir, "mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):

            result_png.save(join(png_save_dir, "%s.1to4png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)

def binarize_image(image,threshold = 127):
    gray_image = image.convert('L')
    # 应用二值化操作
    binary_image = gray_image.point(lambda p: 255 if p > threshold else 0)
    return binary_image

def upgrate_label(image_file,image_file1):  #第一个输入4等分和1图像文件夹  第二个输入奇偶拼接图像的文件夹
    for filename in os.listdir(image_file):
        file_path = os.path.join(image_file, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image= Image.open(file_path)
            binary_image = binarize_image(image)
            image = np.array(binary_image)
            mask = iss_edge(image)


        file_path1 = os.path.join(image_file1,filename)
        if os.path.isfile(file_path1) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image1 = Image.open(file_path1)
            binary_image1 = binarize_image(image1)
            image1 = np.array(binary_image1)
            mask1,mask2,mask3,mask4 = iss_zituedge(image1)


            finalmask1 = np.zeros_like(mask, dtype=int)

            finalmask2 = np.zeros_like(mask, dtype=int)

            masks = np.stack([mask, mask1, mask2, mask3, mask4], axis=-1)
            count_ones = np.sum(masks, axis=-1)



            finalmask1[count_ones > 2] = 1
            finalmask1[finalmask1 == 1] = 255
            finalmask1 = np.pad(finalmask1, ((0, 0), (0, 1)), mode='constant')
            finalmask1 = np.pad(finalmask1, ((0, 1), (0, 0)), mode='constant')

            final_image1 = Image.fromarray(finalmask1.astype(np.uint8))
            #final_image1.save('D:\研究生文件/6.UAED\\new_data_file\groundTruth/final_new_label/1th{}'.format(filename))

            finalmask2[count_ones > 2] = 1
            finalmask2[finalmask2 == 1] = 255
            finalmask2 = np.pad(finalmask2, ((0, 0), (0, 1)), mode='constant')
            finalmask2 = np.pad(finalmask2, ((0, 1), (0, 0)), mode='constant')
            final_image2 = Image.fromarray(finalmask2.astype(np.uint8))
            #final_image2.save('D:\研究生文件/6.UAED\\new_data_file\groundTruth/final_new_label/2th{}'.format(filename))
            # final_image2.save('D:\研究生文件/6.UAED\\new_data_file\groundTruth/final_new_label/3th{}'.format(filename))
            # final_image2.save('D:\研究生文件/6.UAED\\new_data_file\groundTruth/final_new_label/4th{}'.format(filename))
def unpad_to_original_size(padded_tensor, original_h, original_w):
    # 检查张量的维度
    if len(padded_tensor.shape) == 4:
        batch_size, channels, padded_h, padded_w = padded_tensor.shape
        original_tensor = padded_tensor[:, :, :original_h, :original_w]
    elif len(padded_tensor.shape) == 3:
        channels, padded_h, padded_w = padded_tensor.shape
        original_tensor = padded_tensor[:, :original_h, :original_w]
    else:
        raise ValueError("Unexpected tensor shape: {}".format(padded_tensor.shape))

    return original_tensor







def get_mask(outputs,image_name):
    _,_,height,width = outputs.shape
    oh=160
    ow=240

    output_splits = torch.unbind(outputs, dim=0)
    img_1 = torch.zeros(oh*2,ow*2)

    #四象限拼接
    top_left = torch.squeeze(output_splits[0])
    bottom_left = torch.squeeze(output_splits[1])
    top_right = torch.squeeze(output_splits[2])
    bottom_right = torch.squeeze(output_splits[3])

    top_left = torch.where(top_left < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    bottom_left = torch.where(bottom_left < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    top_right = torch.where(top_right < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    bottom_right = torch.where(bottom_right < 0.5, torch.tensor(0.0), torch.tensor(1.0))

    top_left = top_left[:oh,:ow]
    bottom_left = bottom_left[:oh,:ow]
    top_right = top_right[:oh,:ow]
    bottom_right = bottom_right[:oh,:ow]

    img_1[:oh, :ow] = top_left  # Top-left
    img_1[oh:, :ow] = bottom_left  # Bottom-left
    img_1[:oh, ow:] = top_right  # Top-right
    img_1[oh:, ow:] = bottom_right  # Bottom-right

    #奇偶拼接
    img_2 = torch.zeros(oh * 2, ow * 2)
    a = torch.squeeze(output_splits[4])
    b = torch.squeeze(output_splits[5])
    c = torch.squeeze(output_splits[6])
    d = torch.squeeze(output_splits[7])

    a = torch.where(a < 0.5,torch.tensor(0.0),  torch.tensor(1.0))
    b = torch.where(b < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    c = torch.where(c < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    d = torch.where(d < 0.5, torch.tensor(0.0), torch.tensor(1.0))

    a = a[:oh,:ow]
    b = b[:oh,:ow]
    c = c[:oh,:ow]
    d = d[:oh,:ow]

    img_2[::2, ::2] = a  # a
    img_2[::2, 1::2] = b  # b
    img_2[1::2, ::2] = c  # c
    img_2[1::2, 1::2] = d # d



    padded_img_2 = torch.zeros(oh* 2 + 2, ow * 2 + 2)
    padded_img_2[1:-1, 1:-1] = img_2
    np.savetxt('e.txt', padded_img_2.detach().cpu().numpy(), fmt='%d')

    edge = torch.tensor(is_edge(img_1.detach().numpy()))

    edge1,edge2,edge3,edge4 = is_zituedge(padded_img_2.detach().numpy())
    edge1 = torch.tensor(edge1)
    edge2 = torch.tensor(edge2)
    edge3 = torch.tensor(edge3)
    edge4 = torch.tensor(edge4)

    e =  torch.zeros(oh*2,ow*2)
    e[:2*oh-1,:2*ow-1] = edge


    e1 =  torch.zeros(oh*2,ow*2)
    e1[:2*oh-1,:2*ow-1] = edge1

    e2 =  torch.zeros(oh * 2, ow * 2)
    e2[:2*oh-1,:2*ow-1] = edge2
    e3 =  torch.zeros(oh * 2, ow * 2)
    e3[:2*oh-1,:2*ow-1] = edge3

    e4 =  torch.zeros(oh * 2, ow * 2)
    e4[:2*oh-1,:2*ow-1] = edge4


    #mask = (e == 1).long() + (e1 == 1).long() + (e2 == 1).long() + (e3 == 1).long() + (e4 == 1).long()
    mask =get_cv_mask(e,e1,e2,e3,e4)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, f"{image_name}.txt")

    # np.savetxt(save_path, mask.detach().cpu().numpy(), fmt='%d')
    # print(f"Mask saved to {save_path}")
    top_left, bottom_left, top_right, bottom_right = split_array_into_quarters(mask)
    odd_rows_odd_cols,odd_rows_even_cols,even_rows_odd_cols,even_rows_even_cols = split_array_into_oddeven(mask)

    top_left,_ = pad_to_multiple_of_eight(top_left)
    bottom_left,_ = pad_to_multiple_of_eight(bottom_left)
    top_right,_ = pad_to_multiple_of_eight(top_right)
    bottom_right,_ = pad_to_multiple_of_eight(bottom_right)
    even_rows_odd_cols,_ = pad_to_multiple_of_eight(even_rows_odd_cols)
    odd_rows_odd_cols,_ = pad_to_multiple_of_eight(odd_rows_odd_cols)
    odd_rows_even_cols,_ = pad_to_multiple_of_eight(odd_rows_even_cols)
    even_rows_even_cols,_ = pad_to_multiple_of_eight(even_rows_even_cols)

    loss_mask = combine_arrays_into_tensor(top_left,bottom_left,top_right,bottom_right,odd_rows_odd_cols,odd_rows_even_cols,even_rows_odd_cols,even_rows_even_cols)


    return loss_mask
def pad_to_multiple_of_eight(tensor):
    # print("tensor.shape: ", tensor.shape)
    tensor = tensor.float() if tensor.dtype != torch.float32 else tensor
    h, w = tensor.shape
    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32
    # 在 PyTorch 1.7 中，可能需要分两步进行填充
    # 首先填充宽度维度
    tensor = torch.nn.functional.pad(tensor, (0, w_pad), mode='reflect')
    # 然后填充高度维度
    if h_pad > 0:
        # 需要一个额外的维度来处理高度填充
        new_shape = list(tensor.shape)
        new_shape[-2] += h_pad  # 增加高度维度
        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        new_tensor[:, :tensor.shape[1], :] = tensor
        tensor = new_tensor
    return tensor, (h, w, h_pad, w_pad)

def get_cv_mask(e, e1, e2, e3, e4):
    # 将 e1, e2, e3, e4 堆叠成一个张量
    e_stack = torch.stack([e1, e2, e3, e4], dim=0)  # 形状: [4, height, width]

    # 统计 e1, e2, e3, e4 中为 1 的数量
    count_ones = e_stack.sum(dim=0)  # 形状: [height, width]

    # 统计 e1, e2, e3, e4 中为 0 的数量
    count_zeros = 4 - count_ones  # 形状: [height, width]

    # 初始化 mask
    mask = torch.zeros_like(e, dtype=torch.int)

    # 根据 e 的值和 count_ones/count_zeros 的值，为 mask 赋值
    mask[(e == 1) & (count_ones == 4)] = 5
    mask[(e == 0) & (count_zeros == 4)] = 5

    mask[(e == 1) & (count_ones == 3)] = 4
    mask[(e == 0) & (count_zeros == 3)] = 4

    mask[(e == 1) & (count_ones == 2)] = 3
    mask[(e == 0) & (count_zeros == 2)] = 3

    mask[(e == 1) & (count_ones == 1)] = 2
    mask[(e == 0) & (count_zeros == 1)] = 2

    mask[(e == 1) & (count_ones == 0)] = 1
    mask[(e == 0) & (count_zeros == 0)] = 1

    return mask

def get_all_ep_mask_image_paths(folder_path, file_extension=".png"):
    """
    获取指定文件夹中所有符合条件的图片路径，并按文件名排序返回。

    参数:
        folder_path (str): 存储 ep_mask 图片的文件夹路径。
        file_extension (str): 图片文件的扩展名（默认是 .png）。

    返回:
        List[str]: 包含所有符合条件的图片路径的列表，按文件名排序。
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"指定的文件夹路径不存在: {folder_path}")

    # 获取所有指定扩展名的文件
    image_paths = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith(file_extension)
    ]

    # 按文件名排序（确保顺序一致）
    image_paths.sort()

    return image_paths
if __name__ == '__main__':
    # print("Waiting for 3 hours before running main()...")
    # time.sleep(2 * 60 * 60)  # 3 小时 = 3 * 60 分钟 * 60 秒
    main()

# iteration3用a1生成a1+a2,然后去做nms