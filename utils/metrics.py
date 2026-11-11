import torch

def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """
    pred_mask and true_mask should be binary (0 and 1) tensors 
    of the same shape (B, 1, H, W)
    """
    # Flatten to (Batch, -1)
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    true_flat = true_mask.view(true_mask.size(0), -1)

    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection
    
    # IoU = Intersection / Union
    iou = (intersection + smooth) / (union + smooth)
    
    # Dice = 2 * Intersection / (Area1 + Area2)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + true_flat.sum(dim=1) + smooth)
    
    return iou.mean().item(), dice.mean().item()