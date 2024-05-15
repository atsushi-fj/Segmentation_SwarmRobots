import torch
from tqdm.auto import tqdm
import wandb
from utils import save_model


def train_step(model, 
               dataloader,
               loss_fn, 
               optimizer,
               device):
    model.train()
    train_loss = 0
    for batch, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        
        loss = loss_fn(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() 
        
    train_loss = train_loss / len(dataloader)
    return loss


def val_step(model, 
              dataloader,
              loss_fn,
              device):
    model.eval() 
    val_loss = 0
    
    with torch.inference_mode():
        for batch, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    return val_loss


def train(model, 
          train_dataloader, 
          test_dataloader, 
          loss_fn,
          optimizer,
          epochs,
          earlystopping,
          model_name,
          model_dir,
          device):
 
    results = {"train_loss": [],
               "test_loss": [],
    }
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)
    
        test_loss = val_step(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device)
        wandb.log({"Epoch": epoch+1,
                   "train_loss": train_loss,
                   "test_loss": test_loss})
        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        
        earlystopping(test_loss, model, model_dir, model_name)
        if earlystopping.early_stop: 
            print("Early Stopping!")
            break
        
        
def train_for_submit(model, 
                    train_dataloader, 
                    optimizer,
                    epochs,
                    model_name,
                    device):
 
    results = {"train_loss": []}
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                device=device)
    
        wandb.log({"Epoch": epoch+1,
                   "train_loss": train_loss})
        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
        )
        
        save_model(model=model,
                   target_dir="models",
                   model_name=model_name)

        results["train_loss"].append(train_loss)
        
