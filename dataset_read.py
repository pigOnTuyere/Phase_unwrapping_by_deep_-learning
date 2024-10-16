import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
from PIL import Image

# 1. 输入数据条纹图像，标签是mat数据，
class PhaseDataset(Dataset):
    def __init__(self, img_dir, dir_gt,transform,target_transform, extension='.mat'):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.dir_gt = dir_gt
        self.extension = extension
        self.transform = transform
        self.target_transform = target_transform


    ' Ask for input and ground truth'
    def __getitem__(self, index):
        # Get an ID of the input and ground truth
        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = Image.open(img_path)
        #
        base_name = self.img_list[index].rsplit(".", 1)[0]
        gt_path_sin = os.path.join(self.dir_gt, base_name + "_sin" + self.extension)
        gt_path_cos = os.path.join(self.dir_gt, base_name + "_cos" + self.extension)

        gt_sin = sio.loadmat(gt_path_sin)["SIN"]
        gt_cos = sio.loadmat(gt_path_cos)["COS"]
        # Open them

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt_sin = self.target_transform(gt_sin)
            gt_cos = self.target_transform(gt_cos)
            gt = torch.stack((gt_sin,gt_cos),dim=0)

        return image, gt

    ' Length of the dataset '
    def __len__(self):
        return len(self.img_list)

from torchvision import transforms
def loadData(batchSize):

    imgTrainDir = r"E:\结构光\danmu_nixiangji_duopin\danmu_nixiangji_duopin\data\images"
    gtTrainDir = r"E:\结构光\danmu_nixiangji_duopin\danmu_nixiangji_duopin\data\labels"
    transform = {
        "images": transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((128,128)),
                                     transforms.Normalize([0.485], [0.229])], ),
        "labels": transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128)),])
    }

    train_dataset = PhaseDataset(imgTrainDir,gtTrainDir,transform["images"], transform["labels"])

    val_dataset   = PhaseDataset(imgTrainDir,gtTrainDir,transform["images"], transform["labels"])

    train_loader = DataLoader(
        train_dataset, batch_size=batchSize, shuffle=True, num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader,val_loader

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512)),
                                     transforms.Normalize([0.485], [0.229])],)
    target_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(300),])
    imgdir = r"E:\结构光\danmu_nixiangji_duopin\danmu_nixiangji_duopin\data\images"
    gtDir = r"E:\结构光\danmu_nixiangji_duopin\danmu_nixiangji_duopin\data\labels"
    dataSet = PhaseDataset(imgdir,gtDir,transform,target_transform)
    for i in range(dataSet.__len__()):
        img,gt = dataSet.__getitem__(i)
        print(img.size())
        print(gt.size())

