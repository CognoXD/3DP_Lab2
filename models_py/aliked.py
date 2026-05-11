import torch
import cv2
import numpy as np
import glob
import os
from lightglue import ALIKED
from lightglue.utils import load_image, rbd

# Configuration
IMAGE_FOLDER = "./datasets/images_3/" 
OUTPUT_FOLDER = "./datasets/aliked_3/" 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
extensions = ["*.jpg", "*.jpeg", "*.png"]

# Initialize Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# max_num_keypoints=5000 provides a good balance for SfM tasks
extractor = ALIKED(max_num_keypoints=5000).eval().to(device)

def save_features_to_yaml(filename, keypoints, descriptors):
    """Saves keypoints and descriptors as float32 matrices for OpenCV compatibility."""
    kpts_matrix = np.float32(keypoints)
    desc_matrix = np.float32(descriptors)
    
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("keypoints", kpts_matrix)
    fs.write("descriptors", desc_matrix)
    fs.release()

# Find and deduplicate image paths
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext.upper())))

image_paths = sorted(list(set(image_paths)))
if not image_paths:
    print(f"No images found in {IMAGE_FOLDER}!")
    exit()

# Extraction Loop
with torch.no_grad():
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        image_tensor = load_image(img_path).to(device)
        
        # Extract features using LightGlue utilities
        feats = extractor.extract(image_tensor)
        feats = rbd(feats) # Remove batch dimension
        
        kpts = feats["keypoints"].cpu().numpy()
        desc = feats["descriptors"].cpu().numpy()
        
        base_filename = os.path.basename(img_path)
        output_yaml_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.yaml") 
        
        save_features_to_yaml(output_yaml_path, kpts, desc)

print("\nextraction complete!")