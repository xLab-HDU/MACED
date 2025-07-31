from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio.v2 as imageio
import torchvision.transforms as transforms
import scipy.io
import random

def pad_to_multiple_of_eight(tensor):
    # print("tensor.shape: ", tensor.shape)
    _, h, w = tensor.shape
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


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='dataset/bsds/', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train.lst')

        elif self.split == 'test':
            self.filelist = join(self.root, 'train.lst') # test不用变，一直都用这个测
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):

        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]

            label_list = []
            for i_label in range(1, len(img_lb_file)):
                lb_path = os.path.join(self.root, img_lb_file[i_label])
                lb = imageio.imread(lb_path)
                # print("lb.shape", lb.shape)
                label = torch.from_numpy(lb).float() / 255
                label_list.append(label.unsqueeze(0))

            labels = torch.cat(label_list, 0)
            lb_mean, original_size = pad_to_multiple_of_eight(labels.mean(dim=0).unsqueeze(0))


            # lb_std, _ = pad_to_multiple_of_eight(labels.std(dim=0.9_mask_a1+a2.9_mask_a1).unsqueeze(0.9_mask_a1+a2.9_mask_a1)) # 使用方差
            lb_std, _ = pad_to_multiple_of_eight(labels.std(dim=0).unsqueeze(0))

            lb_std_shape = labels.std(dim=0).unsqueeze(0).shape  # 原始形状
            # lb_std = torch.zeros_like(labels.std(dim=0).unsqueeze(0))  # 全 0 张量
            # 如果需要填充到 8 的倍数（保持原逻辑）
            # lb_std, _ = pad_to_multiple_of_eight(lb_std)
            lb_index = random.randint(2, len(img_lb_file)) - 1
            lb_file = img_lb_file[lb_index]
        else:
            img_file = self.filelist[index].rstrip()

        img_path = os.path.join(self.root, img_file)
        img = imageio.imread(img_path)
        img = transforms.ToTensor()(img)
        img, original_size = pad_to_multiple_of_eight(img.float())

        if self.split == "train":
            lb_path = os.path.join(self.root, lb_file)
            lb = imageio.imread(lb_path)
            label = torch.from_numpy(lb).float().unsqueeze(0) / 255
            label, _ = pad_to_multiple_of_eight(label)
            return img, label, lb_mean, lb_std, original_size, img_file
        else:
            return img, img_file

if __name__ == "__main__":
    root = 'E:/UAED/6.UAED/6.UAED\dataset/bsds/'
    dataset = BSDS_RCFLoader(root=root, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    for (img, label, lb_mean, lb_std, original_size, img_file) in train_loader:
        print(type(img), img_file, type(img_file))
        break
        # label_values, counts = torch.unique(label, return_counts=True)
        # print(f"Image file: {img_file}")
        # print("Label values and their counts:")
        # for value, count in zip(label_values, counts):
        #     print(f"Value: {value.item()}, Count: {count.item()}")
        # break

    # for (img, size) in train_loader:
    #     print(type(img))
    #     print(img, size)
    #     break