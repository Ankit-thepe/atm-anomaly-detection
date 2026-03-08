# In slowfast_model.py

import torch
import torch.nn as nn

def SlowFast(num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    
    # --- Unfreeze the last two blocks ---
    for param in model.parameters():
        param.requires_grad = False # First, freeze everything
        
    for param in model.blocks[5].parameters():
        param.requires_grad = True # Then, unfreeze block 5
        
    for param in model.blocks[6].parameters():
        param.requires_grad = True # And unfreeze block 6 (the head)

    # Replace the final layer
    in_features = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Linear(in_features, num_classes)
    
    return model
