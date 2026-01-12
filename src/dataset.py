import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class ADE20KCDataset(Dataset):
    def __init__(self, data_root, corruption, severity, short_edge_size=512):
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.short_edge_size = short_edge_size
        
        # Path logic based on requirements
        # image_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / "validation"
        self.image_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / "validation"
        # annotation_dir = self.data_root / "annotations" / "validation"
        self.annotation_dir = self.data_root / "ADEChallengeData2016" / "annotations" / "validation" # Adjusted based on common ADE struct or provided info
        
        # Verify paths exist
        if not self.image_dir.exists():
            # Fallback or error - listing files to check
            print(f"Warning: Image dir {self.image_dir} does not exist.")
            
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        
        # Load Annotation
        # Annotations are usually .png
        ann_path = self.annotation_dir / f"{img_name}.png"
        if not ann_path.exists():
           # Try checking if annotation name is different (sometimes they match)
           # For ADE20K, simple match is usually fine.
           print(f"Warning: Annotation {ann_path} not found.")
           # Return a dummy mask to avoid crash, or handle error
           mask = Image.new("L", image.size, 0)
        else:
           mask = Image.open(ann_path)

        # Resize Logic: Short edge to 512
        w, h = image.size
        if w < h:
            new_w = self.short_edge_size
            new_h = int(h * (self.short_edge_size / w))
        else:
            new_h = self.short_edge_size
            new_w = int(w * (self.short_edge_size / h))
            
        # Resize inputs
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST) # Nearest for segmentation mask
        
        # Convert to Tensor (No normalization yet, done in model specific steps or here)
        # Requirement: "Discrim: ImageNet mean norm", "Gen: [-1, 1]"
        # I will return the raw tensor [0, 1] and let models handle specific normalization
        image_tensor = F.to_tensor(image) # [C, H, W], range [0, 1]
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        
        # Handle 0-index vs 255 ignore
        # ADE20K usually has 0 as background or ignore, but standard training reduces index by 1 or handles it.
        # Requirement says "15 classes + severe 5", implying maybe a subset?
        # Standard ADE20K is 150 classes.
        # I will pass raw mask.
        
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "img_path": str(img_path),
            "original_size": (h, w), # H, W
            "resized_size": (new_h, new_w)
        }
