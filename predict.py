import torch
import torchvision
from unet import UNet_256
# from unet import val_loader
from unet import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

device = torch.device("cuda:1,2,3" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self,test_images_dir):
        self.testImagesDir = test_images_dir
        self.ids = [splitext(file)[0] for file in listdir(test_images_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def transform(self, image):
        # resize = transforms.Resize(size=(572,572))
        resize = transforms.Resize(size=(256,256))
        image = resize(image)
        image = TF.to_tensor(image)
        return image

    def __getitem__(self, item):
        idx = self.ids[item]
        img_file = glob(self.testImagesDir + idx + '.*')
        image = Image.open(img_file[0])
        x = self.transform(image)
        return {
            'image': x,
        }

def data_loader(test_dir_path):
    dataset = InferenceDataset(test_images_dir=test_dir_path)
    loader = DataLoader(
        dataset, batch_size=1, drop_last=False, num_workers=1
    )
    return loader

def predict_img(net, full_img, device):

    net.eval()
    # print(full_img.shape)
    # img = full_img.unsqueeze(0)
    img = full_img
    # print(img.shape)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        # print(full_img.size[1])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    print('')
    unet_256_Predict = UNet_256()
    # state_dict = torch.load('checkpoints/CP_epoch30.pth')
    state_dict = torch.load('/home/sysadmin/PycharmProjects/UNet_AbhishekThakur/checkpoints_v2/CP_epoch4.pth')
    # state_dict = torch.load('checkpoints_v2/CP_epoch30.pth')
    unet_256_Predict.load_state_dict(state_dict)
    unet_256_Predict.to(device)

    test_images_path = 'inferenceImages/validation/'
    test_data_loader = data_loader(test_images_path)
    i = 0
    for batch in test_data_loader:

        mask = predict_img(unet_256_Predict, batch['image'], device)
        i = i+1
        result = mask_to_image(mask)
        result.save('output_sample'+ str(i)+ '.jpg')
        # result.save('latest_op_sample'+ str(i)+ '.jpg')

    logging.info("Mask saved to {}".format('output_sample.jpg'))

    # with torch.no_grad():
    #     unet_256_Predict.eval()
    #
    #     input_list = []
    #     pred_list = []
    #     true_list = []
    #
    #     # for i, data in tqdm(enumerate(val_loader)):
    #     for i, data in tqdm(enumerate(test_data_loader)):
    #         x, y_true = data
    #         x, y_true = x.to(device), y_true.to(device)
    #
    #         y_pred = unet_256_Predict(x)
    #         y_pred_np = y_pred.detach().cpu().numpy()
    #         # chk = pred_mask.squeeze(0)
    #         # chk2 = chk.permute(1, 2, 0)
    #         # # plt.imshow(pred_mask.detach().cpu().numpy(), cmap='Greys')
    #         plt.imshow(y_pred_np.detach().cpu().numpy())
    #         plt.show()
    #
    #         pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
    #
    #         y_true_np = y_true.detach().cpu().numpy()
    #         true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
    #
    #         x_np = x.detach().cpu().numpy()
    #         input_list.extend([x_np[s] for s in range(x_np.shape[0])])



print('')



