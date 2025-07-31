import os, sys

sys.path.append("/data/private/zhoucaixia/workspace/UD_Edge")
import numpy as np
from PIL import Image
import cv2
import argparse
import time
import torch
import matplotlib

matplotlib.use('Agg')
from data.data_loader_one_random_uncert import BSDS_RCFLoader
from models.sigma_logit_unetpp import Mymodel
from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from torch.distributions import Normal, Independent
from shutil import copyfile
from PIL import Image
from time import time  # Import time to measure execution time
# from tqdm import tqdm  # 导入 tqdm
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--distribution', default="gs", type=str, help='the output distribution')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#test_dataset = BSDS_RCFLoader(root='./data_file/images/1to4test', split="test")
# test_dataset = BSDS_RCFLoader(root='/Users/mac/Downloads/6.UAED/dataset/bsds/', split="test")
# test_loader = DataLoader(
#     test_dataset, batch_size=1,
#     num_workers=0, drop_last=True, shuffle=False)
root = r'E:\UAED\6.UAED\6.UAED\dataset\bsds'
test_dataset = BSDS_RCFLoader(root=root, split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
train_dataset = BSDS_RCFLoader(root=root, split='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# with open('dataset/bsds/test.lst', 'r') as f:
#     test_list = f.readlines()
# test_list = [split(i.rstrip())[1] for i in test_list]
# assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

model_path = r"E:\UAED\6.UAED\6.UAED\Pth\bsds\epoch-4-checkpoint.pth"
model = Mymodel(args)

def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
device = get_default_device()
model.to(device)
print(device)
#model.cuda()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

save_dir = r"E:\UAED\6.UAED\6.UAED\result\bsds\pth-39"
# os.makedirs(os.path.join(save_dir), exist_ok=True)
# os.makedirs(os.path.join(save_dir, 'images', 'train1to4-bai'), exist_ok=True)
# first_ten_batches = list(test_loader)[:]
start_time = time()  # Start timing
for (image, *_, filename) in test_loader:
    print(type(image), type(filename), filename)
    # 去掉 filename 中的扩展名
    filename = os.path.splitext(str(filename[0]))[0]
    image = image.to(device)
    mean, std = model(image)
    _, _, H, W = image.shape
    outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
    outputs = torch.sigmoid(outputs_dist.rsample())
    png = torch.squeeze(1-outputs.detach()).cpu().numpy()
    result = np.zeros((H , W ))
    result[0:, 0:] = png
    result_png = Image.fromarray((result * 255).astype(np.uint8))

    # 保存图像文件和 .mat 文件到指定目录，不包含子目录
    result_png.save(os.path.join(save_dir, f"{filename}.png"))
    # io.savemat(os.path.join(save_dir, f"{filename}.mat"), {'result': result}, do_compression=True)
end_time = time()  # End timing

elapsed_time = end_time - start_time
print(len(test_loader))
fps = len(test_loader) / elapsed_time
print(f"Average FPS: {fps:.2f}")  # Print the average FPS

# 要当作是test传