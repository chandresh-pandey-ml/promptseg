import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from peft import PeftModel
import numpy as np
import cv2
from tqdm import tqdm
from data.dataloader import test_dataset, val_dataset, taping_train, taping_val
from utils.metrics import calculate_metrics

# === CONFIGURATION ===
"""
Global configuration for evaluation script.
Sets device, model paths, and threshold values for all evaluations.
"""
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device being used : {DEVICE}")
BASE_MODEL_ID = "CIDAS/clipseg-rd64-refined"  # Base CLIPSeg model for zero-shot evaluation
LORA_ADAPTER_PATH = "data/joint_lora_clipseg"  # Path to fine-tuned LoRA adapter
FULL_FT_MODEL_PATH = "data/fine_tuned_clipseg_cracks"  # Path to fully fine-tuned model
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Threshold values for binarizing probability masks

def clean_binary_mask(mask_tensor, kernel_size=3):
    """
    Applies morphological operations to clean up binary masks.
    
    Args:
        mask_tensor: Binary mask tensor from thresholded probabilities
        kernel_size: Size of morphological kernel (default: 3)
    
    Returns:
        Cleaned binary mask tensor
    """
    # Convert tensor to numpy array for OpenCV operations
    mask_np = mask_tensor.numpy().astype(np.uint8)
    # Create a 3x3 kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Remove small noise using morphological opening
    refined = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    # Fill small holes using morphological closing
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    # Convert back to tensor
    return torch.tensor(refined, dtype=torch.float32)

def run_evaluation_sweep(model, processor, dataset, model_name):
    """
    Performs threshold sweep evaluation on a given model and dataset.
    
    Args:
        model: The trained model to evaluate
        processor: CLIPSeg processor for handling inputs
        dataset: Dataset containing images, masks, and prompts
        model_name: Name/description of the model for logging
    
    Returns:
        Dictionary containing best results summary across all thresholds
    """
    print(f"\n>>> Starting Threshold Sweep: {model_name}")
    model.eval()
    
    best_overall_iou = 0.0
    best_summary = {}
    
    all_logits, all_targets, all_prompts = [], [], []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Inference ({model_name})"):
            # Load image, ground truth mask, and prompt text
            img, true_mask, prompt = dataset[i]
            
            # Prepare inputs for the model
            inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            
            # Extract and resize prediction logits to match mask dimensions
            raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
            pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False).squeeze()
            
            # Store predictions (sigmoid applied to get probabilities), targets, and prompts
            all_logits.append(torch.sigmoid(pred_resized).cpu())
            all_targets.append(torch.tensor(true_mask / 255.0, dtype=torch.float32))
            all_prompts.append(prompt)

    # Evaluate predictions at different thresholds
    for thresh in THRESHOLDS:
        current_ious, current_dices = [], []
        prompt_metrics = {}
        
        # Calculate metrics for each image in the dataset
        for i in range(len(all_logits)):
            # Binarize predictions using current threshold
            binary_pred = (all_logits[i] > thresh).float()
            # Clean binary predictions using morphological operations
            clean_pred = clean_binary_mask(binary_pred)
            # Calculate IoU and Dice metrics
            iou, dice = calculate_metrics(clean_pred, all_targets[i])
            
            current_ious.append(iou)
            current_dices.append(dice)
            
            # Track metrics per prompt for detailed analysis
            p = all_prompts[i]
            if p not in prompt_metrics:
                prompt_metrics[p] = {"iou": [], "dice": []}
            prompt_metrics[p]["iou"].append(iou)
            prompt_metrics[p]["dice"].append(dice)

        # Calculate mean metrics across all images for this threshold
        mIoU = np.mean(current_ious)
        mDice = np.mean(current_dices)
        print(f"  Thresh {thresh} -> mIoU: {mIoU:.4f} | Dice: {mDice:.4f}")

        # Keep track of the best threshold found
        if mIoU > best_overall_iou:
            best_overall_iou = mIoU
            best_summary = {
                "threshold": thresh, "iou": mIoU, "dice": mDice,
                "prompt_wise": {p: {"iou": np.mean(m["iou"]), "dice": np.mean(m["dice"])} for p, m in prompt_metrics.items()}
            }

    # Print detailed results for the best threshold
    print(f"\n--- BEST {model_name} RESULTS (Thresh: {best_summary['threshold']}) ---")
    print(f"Overall mIoU: {best_summary['iou']:.4f} | Overall Dice: {best_summary['dice']:.4f}")
    for p, m in best_summary['prompt_wise'].items():
        print(f"  > '{p}': mIoU: {m['iou']:.4f} | Dice: {m['dice']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIPSeg variants on Drywall/Cracks datasets.")
    parser.add_argument("--model", type=str, choices=["base", "lora", "full", "all"], default="all", 
                        help="Which model version to evaluate.")
    parser.add_argument("--dataset", type=str, choices=["cracks_test", "cracks_val", "taping_val", "taping_train"], 
                        default="cracks_test", help="Which dataset split to evaluate on.")
    
    args = parser.parse_args()

    dataset_map = {
        "cracks_test": test_dataset,
        "cracks_val": val_dataset,
        "taping_val": taping_val,
        "taping_train": taping_train
    }
    selected_dataset = dataset_map[args.dataset]
    
    processor = CLIPSegProcessor.from_pretrained(BASE_MODEL_ID)

    if args.model in ["base", "all"]:
        model = CLIPSegForImageSegmentation.from_pretrained(BASE_MODEL_ID).to(DEVICE)
        run_evaluation_sweep(model, processor, selected_dataset, "ZERO-SHOT BASELINE")
        del model

    if args.model in ["lora", "all"]:
        base = CLIPSegForImageSegmentation.from_pretrained(BASE_MODEL_ID)
        model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH).to(DEVICE)
        run_evaluation_sweep(model, processor, selected_dataset, "JOINT LoRA (PEFT)")
        del model, base

    if args.model in ["full", "all"]:
        try:
            model = CLIPSegForImageSegmentation.from_pretrained(FULL_FT_MODEL_PATH).to(DEVICE)
            run_evaluation_sweep(model, processor, selected_dataset, "FULL FINE-TUNING")
            del model
        except Exception as e:
            print(f"Could not load Full FT model: {e}")