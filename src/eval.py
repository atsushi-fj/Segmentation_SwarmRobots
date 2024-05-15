import numpy as np
import argparse
import torch 
import cv2
import matplotlib.pyplot as plt

from utils import load_config, seed_everything
from models.deeplabv3 import DeepLabV3


def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="calculate iou with GradCam results")
    
    parser.add_argument("-config", type=str, default="config1.yaml",
                        help="Set config file")
    
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    
    model = DeepLabV3(encoder_name=cfg["encoder_name"],
                     encoder_weights=cfg["encoder_weights"],
                            encoder_depth=cfg["encoder_depth"],
                            in_channels=cfg["in_channels"],
                            decoder_channels=cfg["decoder_channels"],
                            classes=cfg["n_classes"],
                            activation=cfg["activation"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    seed_everything(seed=cfg["seed"])
    mean = np.array(cfg["mean"])
    std = np.array(cfg["std"])
        
    model.load_state_dict(torch.load("weights/deeplabv3_model_config9.pth"))
    model.eval()
    model.to(device)
    iou = 0
    n_file = 2997
    
    for i in range(1, n_file):
        img_path = f"../input/DNE_3/img_3/{i}.png"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        binary_path = f"../input/DNE_3/binary_3/{i}.png"
        binary_img = cv2.imread(binary_path)
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        binary_img = binary_img // 255
            
        img = img2tensor(img/255.)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model(img)
        segmented_img = torch.argmax(output, dim=1)
        segmented_img = segmented_img.cpu().squeeze(0)
        segmented_img = segmented_img.numpy()
        segmented_img_save = segmented_img * 255
        cv2.imwrite(f"../input/segmented_DNE/{i}.png", segmented_img_save)
        
        eval_img = segmented_img * binary_img
        
        true_positive = np.count_nonzero(eval_img)
        positive = np.count_nonzero(binary_img)
        true = np.count_nonzero(segmented_img)
        iou += true_positive / (positive + true - true_positive + 1e-10)
    miou = iou / n_file
    print(miou)
    
        
        
    
    
    
    
    
    