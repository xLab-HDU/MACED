import os
import sys

import torch


class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

class Averagvalue(object):
    """Computes and stores the average and current value"""

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

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


import numpy as np
from PIL import Image

import numpy as np
from PIL import Image


def count_grayscale_values(image_path):
    # 打开图像并转换为灰度模式
    # img = Image.open(image_path).convert('L')
    img = Image.open(image_path)

    # 将图像转换为 numpy 数组
    img_array = np.array(img)

    # 打印图像数组的形状
    print("Image shape:", img_array.shape)

    # 统计每个灰度值的数量
    unique_values, counts = np.unique(img_array, return_counts=True)
    # 8bit 图像会变成255
    # 打印图像中出现的所有灰度值及其数量
    print("Grayscale values and their counts:")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")

def iss_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4

def is_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4

#将奇偶图拼回原图并加一圈像素  输入：图片所在文件夹和输出文件夹
def combine_images_with_border(input_folder, output_folder, border_size=1, border_color='black'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    grouped_files = {}
    # 只读取符合指定后缀的文件
    files = [f for f in os.listdir(input_folder) if f.endswith(('_a.png', '_b.png', '_c.png', '_d.png'))]

    # 根据基本名称组织文件
    for file in files:
        base_name = file.rsplit('_', 1)[0]
        suffix = file[-5]  # 获取后缀字符 'a', 'b', 'c', 或 'd'
        if base_name not in grouped_files:
            grouped_files[base_name] = {}
        grouped_files[base_name][suffix] = file

    # 检查并组合图像
    for base_name, parts in grouped_files.items():
        required_parts = {'a', 'b', 'c', 'd'}
        missing_parts = required_parts - set(parts.keys())
        if not missing_parts:
            imgs = [np.array(Image.open(os.path.join(input_folder, parts[suffix]))) for suffix in sorted(parts)]

            # 检测图像通道数
            channels = imgs[0].shape[2] if len(imgs[0].shape) == 3 else 1
            full_height, full_width = imgs[0].shape[0] * 2, imgs[0].shape[1] * 2
            if channels > 1:
                combined_image = np.zeros((full_height, full_width, channels), dtype=imgs[0].dtype)
            else:
                combined_image = np.zeros((full_height, full_width), dtype=imgs[0].dtype)

            # 组合图像
            combined_image[::2, ::2] = imgs[0]  # a
            combined_image[::2, 1::2] = imgs[1]  # b
            combined_image[1::2, ::2] = imgs[2]  # c
            combined_image[1::2, 1::2] = imgs[3]  # d

            # 转换为Image对象并添加边框
            combined_image_pil = Image.fromarray(combined_image)


            combined_image_with_border = ImageOps.expand(combined_image_pil, border=border_size, fill=border_color)



            # 保存组合后的图像
            output_image_path = os.path.join(output_folder, base_name + '.png')
            combined_image_with_border.save(output_image_path)
            #print(f"Combined image with border saved to {output_image_path}")
        else:
            print(f"Not all parts are available for {base_name}, missing: {', '.join(missing_parts)}")

#将象限图拼回原图  输入：图片所在文件夹和输出文件夹
def combine_fourimages(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    grouped_files = {}
    # 筛选出符合命名规范的图像文件
    files = [f for f in os.listdir(input_folder) if f.endswith(('_1.png', '_2.png', '_3.png', '_4.png'))]

    # 根据基本名称组织文件
    for file in files:
        base_name = file.rsplit('_', 1)[0]
        suffix = file[-5]  # 获取后缀数字 '1', '2', '3', 或 '4'a
        if base_name not in grouped_files:
            grouped_files[base_name] = {}
        grouped_files[base_name][suffix] = file

    # 检查并组合图像
    for base_name, parts in grouped_files.items():
        required_parts = {'1', '2', '3', '4'}
        missing_parts = required_parts - set(parts.keys())
        if not missing_parts:
            # 按顺序加载图像：左上角（1），左下角（2），右上角（3），右下角（4）
            ordered_suffixes = ['1', '2', '3', '4']
            imgs = [np.array(Image.open(os.path.join(input_folder, parts[suffix]))) for suffix in ordered_suffixes]

            # 检测图像通道数
            channels = imgs[0].shape[2] if len(imgs[0].shape) == 3 else 1
            full_height, full_width = imgs[0].shape[0] * 2, imgs[0].shape[1] * 2
            if channels > 1:
                combined_image = np.zeros((full_height, full_width, channels), dtype=imgs[0].dtype)
            else:
                combined_image = np.zeros((full_height, full_width), dtype=imgs[0].dtype)

            # 组合图像
            combined_image[:full_height // 2, :full_width // 2] = imgs[0]  # 1 - 左上角
            combined_image[full_height // 2:, :full_width // 2] = imgs[1]  # 2 - 左下角
            combined_image[:full_height // 2, full_width // 2:] = imgs[2]  # 3 - 右上角
            combined_image[full_height // 2:, full_width // 2:] = imgs[3]  # 4 - 右下角

            # 保存组合后的图像
            output_image_path = os.path.join(output_folder, base_name + '.png')
            Image.fromarray(combined_image).save(output_image_path)
            #print(f"Combined image saved to {output_image_path}")
        else:
            print(f"Not all parts are available for {base_name}, missing: {', '.join(missing_parts)}")

def iss_edge(img_array):  # 判断该像素是否是边缘
    height, width = img_array.shape
    mask = np.zeros((height, width))
    mask[:height, :width] = img_array
    edges = np.zeros((img_array.shape[0]-1, img_array.shape[1]-1), dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]-1):

        for j in range(0, img_array.shape[1]-1):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 1
            else:
                edges[i, j] = 0
    edges_uint8 = edges.astype(np.uint8)
    # edges_image = Image.fromarray(edges_uint8)
    return edges  # 返回的是图片的numpy数组

def is_zituedge(img_array):
    height, width = img_array.shape
    edges1 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges2 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges3 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    edges4 = np.zeros((img_array.shape[0] - 3, img_array.shape[1] - 3), dtype=int)
    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j-1]
            top_right = img_array[i-1, j+1]
            bottom_left = img_array[i+1, j-1]
            bottom_right = img_array[i+1, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges1[i-1, j-1] = 1
            else:
                edges1[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i-1, j]
            top_right = img_array[i+1, j]
            bottom_left = img_array[i-1, j+2]
            bottom_right = img_array[i+1, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges2[i-1, j-1] = 1
            else:
                edges2[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j-1]
            top_right = img_array[i, j+1]
            bottom_left = img_array[i+2, j-1]
            bottom_right = img_array[i+2, j+1]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges3[i-1, j-1] = 1
            else:
                edges3[i-1, j-1] = 0

    for i in range(1,height-2):
        for j in range(1,width-2):
            top_left = img_array[i, j]
            top_right = img_array[i, j+2]
            bottom_left = img_array[i+2, j]
            bottom_right = img_array[i+2, j+2]
            window = [top_left, top_right, bottom_left, bottom_right]
            # 在此处理window中的数据
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 255 and non_zero_count != 4 and non_zero_count != 0:
                edges4[i-1, j-1] = 1
            else:
                edges4[i-1, j-1] = 0
    return edges1,edges2,edges3,edges4

def is_include_edge(img_array):  # 判断该像素围成的2*2区域是否包含边缘
    height, width = img_array.shape
    mask = np.zeros((height + 1, width + 1))
    mask[:height, :width] = img_array
    edges = np.zeros_like(img_array, dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]):

        for j in range(0, img_array.shape[1]):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 255
    return edges  # 返回的是以该像素为左上角的2*2区域是否包含边缘

# 将图片按奇数行奇数列、奇数行偶数列、偶数行奇数列、偶数行偶数列进行分割
# input_folder为图片所在文件夹，lst_file_path为lst文件,output+folder为输出文件夹
def split_image_into_quadrants(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, image_name)

            img = Image.open(image_path)
            # 将图像转化为数组
            img_array = np.array(img)
            # 彩色图和黑白图
            if len(img_array.shape) == 3:
                height, width, _ = img_array.shape
            else:
                height, width = img_array.shape
            # 分割图像
            odd_rows_odd_cols = img_array[::2, ::2]
            odd_rows_even_cols = img_array[::2, 1::2]
            even_rows_odd_cols = img_array[1::2, ::2]
            even_rows_even_cols = img_array[1::2, 1::2]

            # 将numpy数组转化为图像
            img_odd_odd = Image.fromarray(odd_rows_odd_cols)
            img_odd_even = Image.fromarray(odd_rows_even_cols)
            img_even_odd = Image.fromarray(even_rows_odd_cols)
            img_even_even = Image.fromarray(even_rows_even_cols)

            # 保存分割后的图像
            base_name, ext = os.path.splitext(image_name)
            img_odd_odd.save(os.path.join(output_folder, f"{base_name}_a{ext}"))
            img_odd_even.save(os.path.join(output_folder, f"{base_name}_b{ext}"))
            img_even_odd.save(os.path.join(output_folder, f"{base_name}_c{ext}"))
            img_even_even.save(os.path.join(output_folder, f"{base_name}_d{ext}"))

#奇偶分四个数组
def split_array_into_oddeven(img_array):

    odd_rows_odd_cols = img_array[::2, ::2]
    odd_rows_even_cols = img_array[::2, 1::2]
    even_rows_odd_cols = img_array[1::2, ::2]
    even_rows_even_cols = img_array[1::2, 1::2]

    return odd_rows_odd_cols,odd_rows_even_cols,even_rows_odd_cols,even_rows_even_cols

def split_array_into_quarters(array):
    # 获取数组的行数和列数
    rows, cols = array.shape

    # 计算行和列的中点
    mid_row = rows // 2
    mid_col = cols // 2

    # 分割数组
    top_left = array[:mid_row, :mid_col]
    bottom_left = array[mid_row:, :mid_col]
    top_right = array[:mid_row, mid_col:]
    bottom_right = array[mid_row:, mid_col:]

    return top_left, bottom_left, top_right, bottom_right

def combine_arrays_into_tensor(*arrays):
    # 检查输入数组的数量
    if len(arrays) != 8:
        raise ValueError("必须提供8个数组")

    # 检查每个数组的形状
    for array in arrays:
        # if array.shape != (384, 640):
        if array.shape != (224, 288):
            raise ValueError("每个数组的形状必须为(160, 256)")

    # 将数组添加新的轴，变成(1, 160, 256)的形状
    reshaped_arrays = [array[np.newaxis, :, :] for array in arrays]

    # 将这些数组组合成一个张量，形状为(8, 1, 160, 256)
    tensor = np.stack(reshaped_arrays, axis=0)

    return tensor

def is_edge(img_array):  # 判断该像素是否是边缘
    height, width = img_array.shape
    mask = np.zeros((height, width))
    mask[:height, :width] = img_array
    edges = np.zeros((img_array.shape[0]-1, img_array.shape[1]-1), dtype=int)
    # print(edges)
    for i in range(0, img_array.shape[0]-1):

        for j in range(0, img_array.shape[1]-1):
            window = mask[i:i + 2, j:j + 2]

            # 计算2x2区域内非零元素的数量
            non_zero_count = np.count_nonzero(window)
            # 如果当前像素为1且2x2区域内有0和1，则标记为边缘
            if img_array[i, j] == 1 and non_zero_count != 4 and non_zero_count != 0:
                edges[i, j] = 1
            else:
                edges[i, j] = 0
    edges_uint8 = edges.astype(np.uint8)
    # edges_image = Image.fromarray(edges_uint8)
    return edges  # 返回的是图片的numpy数组


if __name__ == '__main__':
    # 示例调用
    image_path = '/Users/mac/Downloads/6.UAED/bai_81871729339055_.pic.jpg'
    count_grayscale_values(image_path)


