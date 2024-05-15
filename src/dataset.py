import numpy as np
import torch 
import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF


class SegmentDataset(torch.utils.data.Dataset):
    
    def __init__(self, imgs, masks, mean, std, transforms):
        self.imgs = imgs
        self.masks = masks
        self.mean = mean
        self.std = std
        self.transforms = transforms
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img, mask = img2tensor(img/255.0), torch.squeeze(img2tensor(mask, np.long))
        
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask
        
def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def get_train_transforms(image, mask):
    i, j, h, w = transforms.RandomCrop.get_params(image, (128, 128))
        
    image = TF.crop(image,i,j,h,w)
    mask  = TF.crop(mask,i,j,h,w)

    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    return image, mask

def get_val_transforms(image, mask):
    image = TF.center_crop(image,[128,128])
    mask  = TF.center_crop(mask,[128,128])

    return image, mask