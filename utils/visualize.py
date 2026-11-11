from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt
import torch
from data.dataloader import test_dataset,train_dataset

img, mask, prompt = test_dataset[3]
print(f"Loaded Prompt: {prompt}")
print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")

print("Loading CLIPSeg Processor and Model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
print("Successfully loaded!")

inputs = processor(text = prompt, images = [img], padding = "max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
preds = outputs.logits.unsqueeze(0)


plt.figure(figsize=(20, 5))

# Plot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Plot 2: Ground Truth Mask
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

# Plot 3: CLIPSeg Prediction
plt.subplot(1, 3, 3)
plt.imshow(torch.sigmoid(preds[0]).squeeze().numpy(), cmap="viridis") 
plt.title(f"Prediction: '{prompt}'")
plt.axis("off")
plt.tight_layout()
plt.savefig('prediction_visualization.png', dpi=300)
plt.show()