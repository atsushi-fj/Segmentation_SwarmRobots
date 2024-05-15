import os
import cv2
import numpy as np
from skimage.transform import resize
import imageio
import argparse
import copy


class FrameGenerator:
    
    def __init__(self, background_path, swarm_path, dataset_path):
        self.embedding_dim = 128
        self.background_path = background_path
        self.swarm_path = swarm_path
        self.dataset_path = dataset_path
        self.img_dir_path = f"{dataset_path}/imgs"
        self.mask_rgb_dir_path = f"{dataset_path}/masks_rgb"
        self.mask_dir_path = f"{dataset_path}/masks"
        
        self.label_colors = {
                "robot": [255, 255, 255],
            }
        
        self.label_classes = {
            "robot": 1,
        }
        
        self.background_imgs = []
        self.swarm_imgs = []
        self.swarm_labels = []
        for img_id in os.listdir(self.background_path):
            img_path = os.path.join(self.background_path, img_id)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128),
                             interpolation=cv2.INTER_AREA).astype(np.uint8)
            self.background_imgs.append(img)
            
        for swarm_id in sorted(os.listdir(self.swarm_path)):
            img_path = os.path.join(self.swarm_path, swarm_id)
            img = imageio.imread(img_path)  # read image file with alpha channel
            img = cv2.resize(img, (30, 30),
                                   interpolation=cv2.INTER_AREA).astype(np.uint8)
            label_name = swarm_id[:-6]
            self.swarm_labels.append(label_name)  # extract label name from image path
            self.swarm_imgs.append(img)
            
    def _embed_image(self, embedded_img, embedded_label):
        h, w, _ = embedded_img.shape
        scale = 0.5 + np.random.random()
        new_height = int(h * scale)
        new_width = int(w * scale)
        obj = resize(embedded_img,
                     (new_height, new_width),
                     preserve_range=True).astype(np.uint8)
        
        # maybe flip 
        if np.random.random() < 0.5:
            obj = np.fliplr(obj)
        
        # choose a random location to store the object
        row0 = np.random.randint(self.embedding_dim - new_height)
        col0 = np.random.randint(self.embedding_dim - new_width)
        row1 = row0 + new_height
        col1 = col0 + new_width
        
        alpha_mask = (obj[:,:,3] == 0)  # alpha channel
        bg_slice = self.img[row0:row1,col0:col1,:]
        bg_slice = np.expand_dims(alpha_mask, -1) * bg_slice
        bg_slice += obj[:,:,:3]
        self.img[row0:row1, col0:col1, :] = bg_slice
        
        obj_px = np.ones((row1-row0, col1-col0)) * alpha_mask
        obj_rgb = np.stack((obj_px,)*3, axis=-1)
        c_channels = self.label_colors[embedded_label]
        self.mask_rgb[row0:row1,col0:col1,:] = self.mask_rgb[row0:row1,col0:col1,:] * obj_rgb
        for j, c in enumerate(c_channels):
            obj_rgb[:,:,j] = np.where(obj_rgb[:,:,j]==0, c, 0)
        self.mask_rgb[row0:row1,col0:col1,:] = self.mask_rgb[row0:row1,col0:col1,:] + obj_rgb
        
        c_channel = self.label_classes[embedded_label]
        self.mask[row0:row1,col0:col1,0] = self.mask[row0:row1,col0:col1,0] * obj_px
        obj_px = np.where(obj_px==0, c_channel, 0)
        self.mask[row0:row1,col0:col1,0] = self.mask[row0:row1,col0:col1,0] + obj_px
            
            
    def generate_frame(self, n_data, n_swarm=4):
        for n in range(n_data):
            bg_idx = np.random.choice(len(self.background_imgs))
            temp_img = self.background_imgs[bg_idx]
            self.img = copy.deepcopy(temp_img)
            img_h, img_w, _ = self.img.shape
            self.mask_rgb = np.zeros((img_h, img_w, 3))
            self.mask = np.zeros((img_h, img_w, 1))
            
            # add swarm image
            swarm_count = np.random.choice(np.arange(n_swarm+1))
            swarm_idxs = np.random.choice(len(self.swarm_imgs), swarm_count)
            for swarm_idx in swarm_idxs:
                swarm_img_ = self.swarm_imgs[swarm_idx]
                h, _, _ = swarm_img_.shape
                swarm_label = self.swarm_labels[swarm_idx]
                if np.random.random() < 0.5:
                    swarm_img = swarm_img_[0:round(h/2), :, :]
                else:
                    swarm_img = swarm_img_
                self._embed_image(embedded_img=swarm_img,
                              embedded_label=swarm_label,)
                
            if n == 0:
                if os.path.isdir(self.img_dir_path) or os.path.isdir(self.mask_rgb_dir_path)\
                    or os.path.isdir(self.mask_dir_path):
                    print(f"{self.img_dir_path} or {self.mask_rgb_dir_path} or {self.mask_dir_path} directory exists.")
                else:
                    print(f"Did not find {self.img_dir_path} & {self.mask_rgb_dir_path} & {self.mask_dir_path} directory, creating them...")
                    os.mkdir(self.img_dir_path)
                    os.mkdir(self.mask_rgb_dir_path)
                    os.mkdir(self.mask_dir_path)
            cv2.imwrite(self.img_dir_path + f"/img_{n}.png", cv2.cvtColor(self.img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.mask_rgb_dir_path + f"/mask_rgb_{n}.png", cv2.cvtColor(self.mask_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.mask_dir_path + f"/mask_{n}.png", cv2.cvtColor(self.mask.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            if n % 1000 == 0:
                print(f"create {n} images...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate data frames")
    
    parser.add_argument("--dataset-path", type=str, default="../input/train",
                        help="Set dataset location")
    parser.add_argument("-n-data", type=int, default=10,
                        help="Set number of data")
    parser.add_argument("-n-env", type=int, default=2,
                       help="Set number of environments")
    parser.add_argument("-n-swarm", type=int, default=5,
                       help="Set number of swarm robots")
    
    args = parser.parse_args()
    frame_generator = FrameGenerator("../input/background", "../input/swarm_2", args.dataset_path)
    frame_generator.generate_frame(args.n_data, args.n_swarm)
    
    