import os
import sys
sys.path.append('./')
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import torch

def edge_detection(image):
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)

        # Apply Canny edge detection
        canny = cv2.Canny(gray, 100, 200)

        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Normalize and concatenate edges
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = np.stack((sobel, canny, laplacian), axis=-1)

        return edges  # shape: [h, w, 3]
    except Exception as e:
        print(f"Error in edge detection: {e}")
        return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

class CigDatasetBinary(Dataset):
    """
    数据描述:
        目录中每个文件夹都代表一个区域, 每个区域中有real和fake两个文件夹, 分别存放真烟样本和假烟样本, 标签分别为1和0
        binary表示只有两个类别, real和fake
    
    数据预处理:
        Resize: 将所有的数据resize到相同的大小, 例如(224, 224)
        ToTensor: 将数据转换为tensor
        normalize: 将数据进行归一化, 均值为[0.485, 0.456, 0.406], 方差为[0.229, 0.224, 0.225]

    """
    def __init__(self, data_dir, transform_cfg=None, flag='train', img_sizes=(256, 256), return_edge=False):
        """
        Parameters:
            data_dir (str): 数据集所在的目录
            flag (str): 用于指示读取数据集的类型 (train/test/train_val)
            train_val表示使用train的数据和test的transform, 用于避免训练数据的随机性
        """
        self.data_dir = data_dir
        self.flag = "train" if flag == "train_val" else flag
        self.img_size = (img_sizes[0], img_sizes[1]) if len(img_sizes) == 2 else (img_sizes[0], img_sizes[0])
        self.return_edge = return_edge

        # 获取图像路径
        self.img_path_list = self.get_dataset_by_flag("train") if flag == "train_val" else self.get_dataset_by_flag(flag)
        self.img_data_list = []
        self.load_all_data() # 读取数据到内存里面
        
        self.data_trasnform = self.get_transform(transform_cfg["test"]) if flag == "train_val" else self.get_transform(transform_cfg[flag])
    
    def load_all_data(self):
        if len(self.img_data_list) == 0:
            for img_path in self.img_path_list:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_size =img.size
                    if img_size != self.img_size:
                        img = img.resize(self.img_size)
                    self.img_data_list.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        assert len(self.img_data_list) == len(self.img_path_list), "data_list and img_path_list should have the same length"
        print("Load all data finished!")
    
    def get_dataset_by_flag(self, flag):
        """
            根据flag获取数据集
        """
        data_dir = os.path.join(self.data_dir, flag)
        region_list = os.listdir(data_dir)
        img_path_list = []
        for region in region_list:
            region_dir = os.path.join(data_dir, region)
            real_dir = os.path.join(region_dir, 'real')
            fake_dir = os.path.join(region_dir, 'fake')
            real_files = os.listdir(real_dir)
            fake_files = os.listdir(fake_dir)
            for file in real_files:
                img_path_list.append(os.path.join(real_dir, file))
            for file in fake_files:
                img_path_list.append(os.path.join(fake_dir, file))
        return img_path_list

    def get_transform(self, data_transform_cfg):
        """
            获取数据预处理的方法
        """
        data_transform = []
        for key, value in data_transform_cfg.items():
            if hasattr(transforms, key):
                data_transform.append(getattr(transforms, key)(**value))
        return transforms.Compose(data_transform)
    

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = self.img_data_list[index]
        
        if self.return_edge:
            img_np = np.array(img) # shpae: [h, w, 3]
            edge = edge_detection(img_np)
            edge = Image.fromarray(edge)
            edge = self.data_trasnform(edge)

        img = self.data_trasnform(img)
        
        img = torch.cat((img, edge), dim=0) if self.return_edge else img
        
        # label is 1 if the image is real, otherwise 0
        label = 1 if 'real' in img_path else 0
        region = int(img_path.split(os.sep)[-3]) # get region id
        
        return img, label, region
       
class CigDatasetAll(CigDatasetBinary):
    """
    数据描述:
        目录中每个文件夹都代表一个区域, 每个区域中有real和fake两个文件夹, 分别存放真烟样本和假烟样本。
        不同的区域被视为不同的类别, 假设有n个区域, 则有2n个类别(real和fake是两个不同的类别)。
    """
    def __init__(self, data_dir, transform_cfg=None, flag='train', img_sizes=(256, 256), return_edge=False):
        super(CigDatasetAll, self).__init__(data_dir, transform_cfg, flag, img_sizes, return_edge)
        self.region_num = len(os.listdir(os.path.join(data_dir, self.flag))) # get the number of regions
        
    def __getitem__(self, index):
        img, label, region = super(CigDatasetAll, self).__getitem__(index)
        label = label*self.region_num + region # get the label, label = region_id if the image is fake, otherwise label = region_id + region_num
        return img, label, region
          
 
if __name__ == '__main__':
    data_dir = './scan_data_256/1916/images'
    transform_cfg = {
        'train': {
            'ColorJitter': {
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.1,
                'hue': 0.1
            },
            'RandomRotation': {
                'degrees': 10
            },
            'ToTensor': {},
            'Normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'test': {
            'ToTensor': {},
            'Normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }
    dataset = CigDatasetAll(data_dir, transform_cfg, flag='test', return_edge=False)
    for img, label, region in dataset:
        # check nan in img
        assert not torch.isnan(img).any(), "img contains nan"
        print(img.shape, label, region)
    dataset = CigDatasetAll(data_dir, transform_cfg, flag='train', return_edge=False)
    for img, label, region in dataset:
        # check nan in img
        assert not torch.isnan(img).any(), "img contains nan"
        print(img.shape, label, region)