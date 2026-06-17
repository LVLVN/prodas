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

class KolektorDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, debug=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.debug = debug
        
        self.folders = [f for f in os.listdir(self.data_dir) if f.startswith('kos')]
        self.folders.sort()
        
        self.image_files = []
        self.label_files = []
        
        defect_count = 0
        total_count = 0
        
        for folder in self.folders:
            folder_path = self.data_dir / folder
            if not folder_path.is_dir():
                continue
                
            for img_file in folder_path.glob('*.jpg'):
                total_count += 1
                label_file = img_file.parent / f"{img_file.stem}_label.bmp"
                if label_file.exists():
                    try:
                        mask = Image.open(label_file).convert('L')
                        mask_np = np.array(mask)
                        if np.any(mask_np > 0):
                            self.image_files.append(img_file)
                            self.label_files.append(label_file)
                            defect_count += 1
                    except Exception as e:
                        logger.warning(f"Cannot read mask file {label_file}: {e}")
        
        logger.info(f"Found {total_count} images total, {defect_count} with defects")
        
        total_files = len(self.image_files)
        indices = list(range(total_files))
        random.seed(42)
        random.shuffle(indices)
        
        split_idx = int(0.8 * total_files)
        if split == 'train':
            self.image_files = [self.image_files[i] for i in indices[:split_idx]]
            self.label_files = [self.label_files[i] for i in indices[:split_idx]]
        else:
            self.image_files = [self.image_files[i] for i in indices[split_idx:]]
            self.label_files = [self.label_files[i] for i in indices[split_idx:]]
        
        logger.info(f"Assigned {len(self.image_files)} defect images to {split} set")
        
        self.resize_transform = transforms.Resize((224, 224))
        self.categories = {
            1: "defect"
        }
        self.num_classes = len(self.categories)
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
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        if self.debug and idx < 5:
            logger.info(f"Loading image: {img_path}")
            logger.info(f"Loading label: {label_path}")
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
            
            if self.debug and idx < 5:
                logger.info(f"Original image size: {image.size}")
                logger.info(f"Original mask size: {mask.size}")
                
        except Exception as e:
            logger.error(f"Error loading image or mask: {e}")
            if idx > 0:
                return self[idx-1]
            else:
                return {
                    'image': torch.zeros(3, 224, 224),
                    'bbox': torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32),
                    'label': torch.tensor([1.0], dtype=torch.float32),
                    'mask': torch.zeros(1, 224, 224),
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
            defect_type='defect',
            object_only=True
        )

        if self.debug and idx < 5:
            logger.info(f"Generated description: {description}")
        
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)
        
        image = self.normalize(image)
        mask = transforms.ToTensor()(mask)
        
        mask = (mask > 0.5).float()
        
        bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        
        label = torch.tensor([1.0], dtype=torch.float32)
        
        w_scale = 224 / orig_w
        h_scale = 224 / orig_h
        scale_factor = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=torch.float32)
        
        return {
            'image': image,
            'bbox': bbox,
            'label': label,
            'mask': mask,
            'image_id': img_path.name,
            'category_id': 1,
            'scale_factor': scale_factor,
            'img_shape': (224, 224),
            'ori_shape': (orig_h, orig_w),
            'description': description
        }

def create_dataloaders(data_dir, batch_size=8, num_workers=4, debug=False):
    logger = logging.getLogger(__name__)
    
    train_dataset = KolektorDataset(
        data_dir=data_dir,
        split='train',
        debug=debug
    )
    
    test_dataset = KolektorDataset(
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
