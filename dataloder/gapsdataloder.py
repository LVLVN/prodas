import torch 
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import logging
import random
from torchvision.transforms import functional as F
from utils import generate_defect_description

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GAPSDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, debug=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.debug = debug
        
        self.img_dir = self.data_dir / 'croppedimg'
        self.mask_dir = self.data_dir / 'croppedgt'
        
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        if split == 'train':
            self.image_files = [f for f in self.image_files if 'train' in f]
        else:
            self.image_files = [f for f in self.image_files if 'test' in f]
            
        logger.info(f"Found {len(self.image_files)} images for {split} in {self.img_dir}")
        
        if debug and len(self.image_files) > 0:
            logger.info(f"Sample image files: {self.image_files[:5]}")
        
        self.resize_transform = transforms.Resize((224, 224))
        
        self.train_augmentation = None
        if split == 'train':
            self.train_augmentation = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.01,    
                    contrast=0.01,      
                    saturation=0.01,    
                    hue=0.01           
                )
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.categories = {
            1: "PCB defect"
        }
        self.num_classes = len(self.categories)
        
        if debug:
            self._validate_image_mask_pairs()

    def _validate_image_mask_pairs(self):
        logger.info("Validating image-mask pairs...")
        
        missing_masks = []
        for img_file in self.image_files[:min(20, len(self.image_files))]:
            mask_file = img_file.replace('.jpg', '.png')
            mask_path = self.mask_dir / mask_file
            
            if not mask_path.exists():
                missing_masks.append(mask_file)
        
        if missing_masks:
            logger.warning(f"Missing {len(missing_masks)} masks out of {len(self.image_files)} images")
            logger.warning(f"Examples of missing masks: {missing_masks[:5]}")
        else:
            logger.info("All checked image-mask pairs are complete.")

    def add_gaussian_noise(self, img):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img)
        noise = np.random.normal(0., 0.01, img_array.shape)
        noisy_img = img_array + noise * 255
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        
        img_path = self.img_dir / img_filename
        mask_filename = img_filename.replace('.jpg', '.png')
        mask_path = self.mask_dir / mask_filename
        
        if self.debug and idx < 5:
            logger.info(f"Loading image: {img_path}")
            logger.info(f"Loading mask: {mask_path}")
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            if self.debug and idx < 5:
                logger.info(f"Original image size: {image.size}")
                logger.info(f"Original mask size: {mask.size}")
                
                mask_array = np.array(mask)
                unique_values = np.unique(mask_array)
                logger.info(f"Unique values in mask: {unique_values}")
                
                non_zero_ratio = np.sum(mask_array > 0) / (mask_array.shape[0] * mask_array.shape[1])
                logger.info(f"Non-zero pixel ratio in mask: {non_zero_ratio:.4f}")
                
        except Exception as e:
            logger.error(f"Error loading image or mask: {e}")
            if idx > 0:
                return self[idx-1]
            else:
                empty_image = torch.zeros(3, 224, 224)
                empty_mask = torch.zeros(1, 224, 224)
                empty_bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
                empty_label = torch.tensor([1.0], dtype=torch.float32)
                return {
                    'image': empty_image,
                    'bbox': empty_bbox,
                    'label': empty_label,
                    'mask': empty_mask,
                    'image_id': 'error',
                    'category_id': 1,
                    'scale_factor': torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                    'img_shape': (224, 224),
                    'ori_shape': (224, 224),
                    'description': "Error loading image"
                }
        
        orig_w, orig_h = image.size
        
        if self.train_augmentation is not None and self.split == 'train':
            image = self.train_augmentation(image)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        description = generate_defect_description(
            image_np,
            mask_np,
            defect_type='PCB defect',
            prefer_line=True
        )

        if self.debug and idx < 5:
            logger.info(f"Generated description: {description}")
        
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)
        
        image = self.normalize(image)
        mask = transforms.ToTensor()(mask)
        
        bbox = [0.5, 0.5, 1.0, 1.0]
        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        label = torch.tensor([1.0], dtype=torch.float32)
        
        w_scale = 224 / orig_w
        h_scale = 224 / orig_h
        scale_factor = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=torch.float32)
        
        data = {
            'image': image,
            'bbox': bbox,
            'label': label,
            'mask': mask,
            'image_id': img_filename,
            'category_id': 1,
            'scale_factor': scale_factor,
            'img_shape': (224, 224),
            'ori_shape': (orig_h, orig_w),
            'description': description
        }
        
        return data

def create_dataloaders(data_dir, batch_size=8, num_workers=4, debug=False):
    logger = logging.getLogger(__name__)
    
    train_dataset = GAPSDataset(
        data_dir=data_dir,
        split='train',
        debug=debug
    )
    
    test_dataset = GAPSDataset(
        data_dir=data_dir,
        split='test',
        debug=debug
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    if debug and len(train_dataset) > 0:
        logger.info("Getting first training sample for debugging...")
        sample = train_dataset[0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Image shape: {sample['image'].shape}")
        logger.info(f"Mask shape: {sample['mask'].shape}")
        logger.info(f"Description: {sample['description']}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader
