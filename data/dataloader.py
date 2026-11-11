import os
import cv2
from torch.utils.data import Dataset
import random

class DrywallCrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        image_dir: Path to the folder with original .jpg images
        mask_dir: Path to the folder with our generated .png masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.mask_filenames = os.listdir(mask_dir) 

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask_filename = self.mask_filenames[idx]
        parts = mask_filename.split('__')
        base_id = parts[0]
        prompt_part = parts[1] 
        text_prompt = prompt_part.replace(".png","")
        text_prompt = text_prompt.replace("_", " ")
        img_path = os.path.join(self.image_dir, f"{base_id}.jpg")
        full_mask_path = os.path.join(self.mask_dir, mask_filename)
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(full_mask_path, 0)
        if random.random() > 0.5:
            original_image = cv2.flip(original_image, 1)
            mask_image = cv2.flip(mask_image, 1)
        return original_image, mask_image, text_prompt

class DrywallTapingDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_filenames = os.listdir(mask_dir) 

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask_filename = self.mask_filenames[idx]
        parts = mask_filename.split('__')
        base_id = parts[0]
        prompt_part = parts[1] 
        text_prompt = prompt_part.replace(".png","").replace("_", " ")
        
        img_path = os.path.join(self.image_dir, f"{base_id}.jpg") 
        full_mask_path = os.path.join(self.mask_dir, mask_filename)
        
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(full_mask_path, 0)
        
        if random.random() > 0.5:
            original_image = cv2.flip(original_image, 1)
            mask_image = cv2.flip(mask_image, 1)
            
        return original_image, mask_image, text_prompt


train_dataset = DrywallCrackDataset(image_dir="cracks/train", mask_dir="cracks_mask/train")
val_dataset = DrywallCrackDataset(image_dir="cracks/val", mask_dir="cracks_mask/val")
test_dataset = DrywallCrackDataset(image_dir="cracks/test", mask_dir="cracks_mask/test")

taping_train = DrywallTapingDataset(image_dir="Drywall-Join-Detect/train", mask_dir="drywall_join_detect_mask/train")
taping_val = DrywallTapingDataset(image_dir="Drywall-Join-Detect/valid", mask_dir="drywall_join_detect_mask/val")


