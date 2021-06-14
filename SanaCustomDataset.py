from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from os.path import splitext
from os import listdir
from glob import glob
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import os

class SanathananDataset(Dataset):
    def __init__(self,images_dir):
        self.imagesDir = images_dir
        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def transform(self, image):
        image = TF.to_tensor(image)
        return image

    def __getitem__(self, item):
        idx = self.ids[item]
        img_file = glob(self.imagesDir + idx + '.*')
        image = Image.open(img_file[0])
        x = self.transform(image) # converting to Tensor image
        return {
            'image': x,
            'filename' : img_file,
        }

def convolution_ImageProcessing():
    sanaImageDir = '/home/sysadmin/Ashish/SanaImages/phase/'
    sanaDataset = SanathananDataset(images_dir=sanaImageDir)
    print(sanaDataset)
    # sana_dataLoader = DataLoader(sanaDataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    sana_dataLoader = DataLoader(sanaDataset, batch_size=10, shuffle=False)

    print(len(sana_dataLoader))

    for imageData in sana_dataLoader:
        image = imageData['image']
        image = image.to(device)
        imageName = imageData['filename']
        convolvedImage = mp(image)

if __name__ == '__main__':

    device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    since = time.time()
    convolution_ImageProcessing()
    time_elapsed = time.time() - since
    print('Convolution operation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print()