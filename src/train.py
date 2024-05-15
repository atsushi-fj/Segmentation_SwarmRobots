import numpy as np
import wandb
import argparse
import torch 
from sklearn.model_selection import train_test_split
import glob
import os

from utils import load_config, create_display_name, seed_everything, EarlyStopping
from dataset import SegmentDataset, get_train_transforms, get_val_transforms
from models.deeplabv3 import DeepLabV3
from engine import train
from inference import eval_model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="segment model : kaggle segment tissue")
    
    parser.add_argument("-config", type=str, default="config1.yaml",
                        help="Set config file")
    
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    name = create_display_name(experiment_name=cfg["experiment_name"],
                               model_name=cfg["model_name"],
                               extra=cfg["extra"])

    with wandb.init(project=cfg["project"],
                    name=name,
                    config=cfg):
        
        cfg = wandb.config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        
        seed_everything(seed=cfg.seed)
        mean = np.array(cfg.mean)
        std = np.array(cfg.std)
    
        all_imgs = sorted(glob.glob(cfg.img_path))
        all_masks = sorted(glob.glob(cfg.mask_path))
        all_imgs = np.array(all_imgs)
        all_masks = np.array(all_masks)
        train_img, val_img, train_mask, val_mask = train_test_split(all_imgs, all_masks,
                                                                    test_size=cfg.test_size,
                                                                    random_state=cfg.seed)     
        train_dataset = SegmentDataset(imgs=train_img,
                                        masks=train_mask,
                                        mean=mean,
                                        std=std,
                                        transforms=get_train_transforms,)
        
        val_dataset = SegmentDataset(imgs=val_img,
                                        masks=val_mask,
                                        mean=mean,
                                        std=std,
                                        transforms=get_val_transforms,)
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=cfg.batch_size,
                                                        shuffle=True,
                                                        num_workers=os.cpu_count(),
                                                        drop_last=True,
                                                        pin_memory=True,)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=cfg.batch_size,
                                                        shuffle=False,
                                                        num_workers=os.cpu_count(),
                                                        pin_memory=True,)
        
        model = DeepLabV3(encoder_name=cfg.encoder_name,
                            encoder_weights=cfg.encoder_weights,
                            encoder_depth=cfg.encoder_depth,
                            in_channels=cfg.in_channels,
                            decoder_channels=cfg.decoder_channels,
                            classes=cfg.n_classes,
                            activation=cfg.activation)
    
        model.to(device)
        if cfg.model_freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.model.segmentation_head.parameters():
                param.requires_grad = True
            
        loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.lr,
                                      weight_decay=cfg.weight_decay,
                                      eps=cfg.eps,
                                      betas=cfg.betas,)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)

        train(model, train_dataloader, val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=cfg.epochs,
                earlystopping=earlystopping,
                model_dir=cfg.model_dir,
                model_name=cfg.model_path,
                device=device)
        
        model.load_state_dict(torch.load(f=cfg.load_model_path))
        model.to(device)
        
        result = eval_model(model=model,
                            data_loader=val_dataloader,
                            loss_fn=loss_fn,
                            device=device)
        
        print(f"\n resut:\n{result}")
