import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3(nn.Module):
    
    def __init__(self,
                 encoder_name,
                 encoder_weights,
                 encoder_depth,
                 in_channels,
                 decoder_channels,
                 classes,
                 activation,):
        super().__init__()
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            classes=classes,
            activation=activation,
        )
        
    def forward(self, images):
        
        logits = self.model(images)
        
        return logits  
        
