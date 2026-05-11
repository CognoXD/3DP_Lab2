import torch
import cv2
import numpy as np
import glob
import os

# Configuration
IMAGE_FOLDER = "./datasets/images_3/" 
OUTPUT_FOLDER = "./datasets/xfeat_3/" 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
extensions = ["*.jpg", "*.jpeg", "*.png"]

# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load XFeat from PyTorch Hub
xfeat = torch.hub.load(
    'verlab/accelerated_features', 
    'XFeat', 
    pretrained=True, 
    top_k=2048, 
    trust_repo=True
).eval().to(device)

def save_features_to_yaml(filename, keypoints, descriptors):
    """Saves keypoints and descriptors as OpenCV-compatible float32 matrices."""
    kpts_matrix = np.float32(keypoints)
    desc_matrix = np.float32(descriptors)
    
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("keypoints", kpts_matrix)
    fs.write("descriptors", desc_matrix)
    fs.release()

# Gather and deduplicate image paths
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext.upper())))

image_paths = sorted(list(set(image_paths)))
if not image_paths:
    print(f"No images found in {IMAGE_FOLDER}!")
    exit()

# Feature Extraction Loop
with torch.no_grad():
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare tensor: (1, C, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        # Extract features
        output = xfeat.detectAndCompute(img_tensor, top_k=5000)[0]
        
        # Move to CPU for saving
        kpts = output["keypoints"].cpu().numpy()
        desc = output["descriptors"].cpu().numpy()
        
        base_filename = os.path.basename(img_path)
        output_yaml_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.yaml")
        save_features_to_yaml(output_yaml_path, kpts, desc)

print("\nextraction complete!")