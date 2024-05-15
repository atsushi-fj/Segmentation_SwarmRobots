import torch.nn as nn
from torchvision.transforms import functional as F


class PILToTensor(nn.Module):
    def forward(self, image, mask=None):
        image = F.pil_to_tensor(image)
        return image, mask

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
    
    def forward(self, image, mask=None):
        image = F.convert_image_dtype(image, self.dtype)
        return image, mask