import json
import cv2
import numpy as np
import os
import random
import argparse

def generate_crack_masks(json_path, output_dir, prompts):
    """Reads precise polygon coordinates to generate thin crack masks."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading CRACK annotations from {json_path}...")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    for image_id, img_info in images_info.items():
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                for seg in ann.get('segmentation', []):
                    if len(seg) >= 6: 
                        poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(mask, [poly], 255)
        
        base_id = os.path.splitext(file_name)[0]
        safe_prompt = random.choice(prompts).replace(" ", "_").replace("/", "_")
        mask_path = os.path.join(output_dir, f"{base_id}__{safe_prompt}.png")
        cv2.imwrite(mask_path, mask)
        
    print(f"Success! Generated {len(images_info)} CRACK masks in '{output_dir}'.\n")

def generate_drywall_masks(json_path, output_dir, prompts):
    """Reads bbox coordinates to generate rectangular masks for taping areas."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading DRYWALL annotations from {json_path}...")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    for image_id, img_info in images_info.items():
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                if 'bbox' in ann and len(ann['bbox']) == 4:
                    x, y, w, h = [int(v) for v in ann['bbox']]
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        base_id = os.path.splitext(file_name)[0]
        safe_prompt = random.choice(prompts).replace(" ", "_").replace("/", "_")
        mask_path = os.path.join(output_dir, f"{base_id}__{safe_prompt}.png")
        cv2.imwrite(mask_path, mask)
        
    print(f"Success! Generated {len(images_info)} DRYWALL masks in '{output_dir}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file_path", type=str, help="Path to the JSON file")
    parser.add_argument("output_folder", type=str, help="Base path to the output directory")
    parser.add_argument("--dataset", type=str, required=True, choices=['cracks', 'drywall'], 
                        help="Specify which dataset to generate masks for.")
    parser.add_argument("--split", type=str, required=True, choices=['train','val','test'], 
                        help="Specify the split (train/val/test).")
    args = parser.parse_args()

    crack_prompts = ["segment crack", "segment wall crack"]
    drywall_prompts = ["segment taping area", "segment joint/tape", "segment drywall seam"]

    final_output_path = os.path.join(args.output_folder, args.split)

    if args.dataset == 'cracks':
        generate_crack_masks(args.json_file_path, final_output_path, crack_prompts)
    elif args.dataset == 'drywall':
        generate_drywall_masks(args.json_file_path, final_output_path, drywall_prompts)