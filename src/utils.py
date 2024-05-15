import yaml
import random
import os 
import torch
import numpy as np
import wandb 
from pathlib import Path
from datetime import datetime
import cv2        


class EarlyStopping:
    def __init__(self,
                 patience=5,
                 verbose=False,):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, model_dir, model_name):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model, model_dir, model_name)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model, model_dir, model_name)
            self.counter = 0
    
    def checkpoint(self, val_loss, model, model_dir, model_name):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        save_model(model=model,
                   target_dir=model_dir,
                   model_name=model_name)
        self.val_loss_min = val_loss
        
        
def load_config(file="config1.yaml"):
    """Load config file"""
    config_path = Path("../config/")
    with open(config_path / file, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True


def save_model(model,
               target_dir,
               model_name):
  """Saves a PyTorch model to a target directory."""
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  wandb.save("model_save_path")


def create_display_name(experiment_name,
                        model_name,
                        extra=None):

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        name = f"{timestamp}-{experiment_name}-{model_name}-{extra}"
    else:
        name = f"{timestamp}-{experiment_name}-{model_name}"
    print(f"[INFO] Create wandb saving to {name}")
    return name

def save_img(data,name,out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)