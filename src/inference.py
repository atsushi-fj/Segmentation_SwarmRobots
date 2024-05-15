import torch 
from tqdm import tqdm


def eval_model(model,
               data_loader,
               loss_fn,
               device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss = 0
    model.eval()
    with torch.inference_mode():
        for images, masks in tqdm(data_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss += loss.item()
        loss /= len(data_loader)

    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item()}
