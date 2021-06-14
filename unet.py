import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from glob import glob
from loss_functions import DiceLoss, IoULoss
import logging
import os
# import matplotlib
# matplotlib.use("TkAgg")
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv

def double_conv_256(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    # print(tensor.size())
    # print(tensor_size-delta)
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,kernel_size=2,stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=2,stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=2,stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, image):
        #encoder
        print('Input image size', image.size())
        x1 = self.down_conv_1(image)
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        print(x7.size())
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print('Bottleneck size:',x9.size())

        #decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        # print('cropped image size', y.size())
        x = self.up_conv_1(torch.cat((x, y), 1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat((x, y), 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat((x, y), 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat((x, y), 1))

        x = self.out(x)
        # print(x.shape)
        # print(x.size())
        return x

class UNet_256(nn.Module):
    def __init__(self):
        super(UNet_256, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv_256(1,32)
        self.down_conv_2 = double_conv_256(32,64)
        self.down_conv_3 = double_conv_256(64,128)
        self.down_conv_4 = double_conv_256(128,256)
        self.down_conv_5 = double_conv_256(256,512)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=2,stride=2)
        self.up_conv_1 = double_conv_256(512, 256)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        self.up_conv_2 = double_conv_256(256, 128)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=2,stride=2)
        self.up_conv_3 = double_conv_256(128, 64)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32,kernel_size=2,stride=2)
        self.up_conv_4 = double_conv_256(64, 32)

        self.out = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)


    def forward(self, image):
        #encoder
        # print('Input image size', image.size())
        x1 = self.down_conv_1(image)
        # print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        # print(x7.size())
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # print('Bottleneck size:',x9.size())

        #decoder
        x = self.up_trans_1(x9)
        # print(x.size())
        # print('')

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat((x, y), 1))
        # print(x.size())

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat((x, y), 1))
        # print(x.size())

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat((x, y), 1))

        x = self.out(x)
        # print(x.shape)
        # print(x.size())
        return x

from torch.utils.data import Dataset, DataLoader, random_split
from os.path import splitext
from os import listdir
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from PIL import Image

class UNetDataset(Dataset):
    def __init__(self,images_dir,target_dir, train=True):
        self.imagesDir = images_dir
        self.targetDir = target_dir
        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def transform(self, image, target):
        # resize = transforms.Resize(size=(572,572))
        resize = transforms.Resize(size=(256,256))
        # resize = transforms.Resize(size=(1024,1024))
        image = resize(image)
        target = resize(target)
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)
        return image, target

    def __getitem__(self, item):
        idx = self.ids[item]
        # print(idx)
        mask_file = glob(self.targetDir + idx + '.*')
        img_file = glob(self.imagesDir + idx + '.*')

        image = Image.open(img_file[0])
        target = Image.open(mask_file[0])
        x, y = self.transform(image, target)
        return {
            'image': x,
            'mask': y,
            'filename' : idx,
        }

import numpy as np
import torch
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

# class HausdorffDTLoss(nn.Module):
# """Binary Hausdorff loss based on distance transform"""

@torch.no_grad()
def distance_field(img: np.ndarray) -> np.ndarray:
    field = np.zeros_like(img)

    for batch_size in range(len(img)):
        fg_mask = img[batch_size] > 0.5

        if fg_mask.any():
            bg_mask = ~fg_mask

            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)

            field[batch_size] = fg_dist + bg_dist

    return field

def HausdorffDTLoss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    #     pred_dt = torch.from_numpy(distance_field(y_pred.cpu().numpy())).float()
    #     true_dt = torch.from_numpy(distance_field(y_true.cpu().numpy())).float()
    y_pred = torch.squeeze(y_pred)
    print(y_pred.shape)
    print(type(y_pred))

    pred_dt = distance_field(y_pred.numpy())
    true_dt = distance_field(y_true.numpy())

    pred_error = (y_pred - y_true) ** 2
    distance = pred_dt ** 2 + true_dt ** 2
    print(distance)

    dt_field = pred_error * distance
    print(dt_field)
    #     return dt_field.mean()
    return dt_field


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)

    # image = torch.rand((1,1,572,572))
    # unetModel = UNet()
    # print(unetModel(image))

    # image = torch.rand((1,1,256,256))
    unetModel_256 = UNet_256()
    unetModel_512 = UNet()
    # print(unetModel_256(image))

    val_percent = 0.2
    # batch_size = 10
    batch_size = 1
    # epochs = 50 # original value
    epochs = 5 # just for testing
    lr = 0.001
    dir_images = 'data/imgs/'
    dir_masks = 'data/masks/'
    dir_checkpoint = 'checkpoints_v2/'
    save_cp = True

    dataset = UNetDataset(dir_images, dir_masks)
    print(dataset)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    print(train_loader)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    print(val_loader)

    # device = torch.device("cuda:1,2,3" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0,1,2" if torch.cuda.is_available() else "cpu")

    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(unetModel_256.parameters(), lr=lr)
    unetModel_256.to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    criterion = IoULoss()
    # criterion = nn.CrossEntropyLoss()

    # for epoch in range(epochs):
    #     unetModel_256.train()
    #     epoch_loss = 0

    for epoch in tqdm(range(epochs), total=epochs):
        unetModel_256.train()
        epoch_loss = 0

        optimizer.zero_grad()

        for batch in train_loader:
            image = batch['image']
            mask = batch['mask']
            patientID = batch['filename']

            image = image.to(device)
            # image_disp = image.squeeze(0).permute(1, 2, 0)
            # plt.imshow(image_disp.detach().cpu().numpy())
            # plt.show()

            true_mask = mask.to(device)
            # true_mask_disp = true_mask.squeeze(0).permute(1, 2, 0)
            # plt.imshow(true_mask_disp.detach().cpu().numpy())
            # plt.show()

            pred_mask = unetModel_256(image)
            # pred_mask_disp = pred_mask.squeeze(0).permute(1, 2, 0)
            # # plt.imshow(pred_mask.detach().cpu().numpy(), cmap='Greys')
            # plt.imshow(pred_mask_disp.detach().cpu().numpy())
            # plt.show()

            # saveFileName = 'Epoch_'+str(epoch)+'.jpg'
            loss = criterion(pred_mask, true_mask)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            global_step += 1

        print('Epoch ' + str(epoch) + '  ' + str(epoch_loss))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(unetModel_256.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')




