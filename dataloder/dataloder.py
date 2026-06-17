import torch 
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import logging
import random
from torchvision.transforms import functional as F
import cv2
from utils import generate_defect_description
import math
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image_with_chinese_path(image_path):
    try:
        image_path_str = str(image_path)
        stream = open(image_path_str, 'rb')
        bytes_img = bytearray(stream.read())
        numpyarray = np.asarray(bytes_img, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

class DefectDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, split='train', test_mode=False, few_shot_config=None):
        self.image_dir = Path(image_dir)
        self.transform = None
        self.test_mode = test_mode
        self.split = split
        self.logger = logging.getLogger(__name__)
        
        self.resize_transform = transforms.Resize((224, 224))
        
        self.train_augmentation = None
        if split == 'train':
            self.train_augmentation = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.categories = {
            cat['id']: cat['name']
            for cat in self.annotations['categories']
        }
        self.num_classes = len(self.categories)
        self.logger.info(f"Found {self.num_classes} categories: {list(self.categories.values())}")
        
        self.image_info = {}
        for img in self.annotations['images']:
            self.image_info[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }
        
        self.image_annotations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            
            img_info = self.image_info[image_id]
            category_name = self.categories[ann['category_id']]
            
            image_path = self.image_dir / category_name / img_info['file_name']
            image = load_image_with_chinese_path(image_path)
            if image is None:
                self.logger.warning(f"Failed to load image: {image_path}")
                continue
            
            mask = None
            if 'segmentation' in ann:
                mask = self.create_binary_mask(ann['segmentation'], img_info['height'], img_info['width'])
                mask = (mask.numpy() * 255).astype(np.uint8)
            
            final_desc = generate_defect_description(image, mask=mask, defect_type=category_name)
            if final_desc.startswith("Unable"):
                final_desc = ann.get('description', f"A defect in the fabric")
            
            ann['processed_description'] = final_desc
            self.image_annotations[image_id].append(ann)
        
        self.image_ids = [img_id for img_id in self.image_annotations.keys() 
                         if len(self.image_annotations[img_id]) > 0]
        
        if self.split == 'train' and few_shot_config and few_shot_config.get('enabled', False):
            self.logger.info("Applying few-shot learning modifications to the training set.")
            
            rare_classes_names = few_shot_config.get('rare_classes', [])
            shots = few_shot_config.get('shots', 10)
            
            name_to_id = {v: k for k, v in self.categories.items()}
            rare_category_ids = {name_to_id[name] for name in rare_classes_names if name in name_to_id}
            
            if not rare_category_ids:
                self.logger.warning("Few-shot learning enabled, but no valid rare classes were specified.")
            else:
                self.logger.info(f"Rare classes: {rare_classes_names} (IDs: {rare_category_ids}) with {shots} shots.")
                
                images_by_category = {cat_id: [] for cat_id in self.categories.keys()}
                for image_id in self.image_ids:
                    ann = self.image_annotations[image_id][0]
                    cat_id = ann['category_id']
                    images_by_category[cat_id].append(image_id)
                    
                new_image_ids = []
                for cat_id, img_ids in images_by_category.items():
                    if cat_id in rare_category_ids:
                        if len(img_ids) > shots:
                            sampled_ids = random.sample(img_ids, shots)
                            new_image_ids.extend(sampled_ids)
                            self.logger.info(f"Sampled {len(sampled_ids)} images for rare class '{self.categories[cat_id]}'.")
                        else:
                            new_image_ids.extend(img_ids)
                            self.logger.warning(f"Class '{self.categories[cat_id]}' has only {len(img_ids)} images, less than the required {shots} shots. Using all.")
                    else:
                        new_image_ids.extend(img_ids)
                
                self.image_ids = new_image_ids
                random.shuffle(self.image_ids)
                self.logger.info(f"After few-shot sampling, the training set has {len(self.image_ids)} images.")

        self.logger.info(f"Loaded {len(self.image_ids)} annotated images")
        
        self.category_to_onehot = {}
        for cat_id in self.categories.keys():
            onehot = torch.zeros(self.num_classes)
            onehot[cat_id - 1] = 1
            self.category_to_onehot[cat_id] = onehot

    def convert_bbox_format(self, bbox, orig_w, orig_h):
        x1, y1, w, h = bbox
        
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        w = min(w, orig_w - x1)
        h = min(h, orig_h - y1)
        
        x_center = x1 + w/2
        y_center = y1 + h/2
        
        x_center /= orig_w
        y_center /= orig_h
        w /= orig_w
        h /= orig_h
        
        return [x_center, y_center, w, h]

    def create_binary_mask(self, segmentation, height, width):
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if isinstance(segmentation, list) and len(segmentation) > 0:
            for polygon_points in segmentation:
                polygon = []
                for i in range(0, len(polygon_points), 2):
                    if i + 1 < len(polygon_points):
                        x, y = polygon_points[i], polygon_points[i+1]
                        polygon.append((float(x), float(y)))
                
                if len(polygon) >= 3:
                    draw.polygon(polygon, outline=255, fill=255)
        
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        return torch.tensor(mask_array, dtype=torch.float32)

    def clean_description(self, text):
        unwanted_patterns = [
            "以上翻译结果来自有道神经网络翻译（YNMT）· 通用场景",
            "重点词汇",
            r"\d+/\d+",
            "通用场景"
        ]
        
        cleaned_text = text
        for pattern in unwanted_patterns:
            cleaned_text = cleaned_text.replace(pattern, "")
        
        return cleaned_text.strip()

    def adjust_gamma(self, img):
        return F.adjust_gamma(img, random.uniform(0.8, 1.2))

    def add_gaussian_noise(self, img):
        if random.random() > 0.4:
            return img
            
        img_array = np.array(img)
        noise_level = random.uniform(0.1,0.2)
        noise = np.random.normal(0., noise_level, img_array.shape)
        noisy_img = img_array + noise * 255
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def add_local_noise(self, img, mask):
        img_array = np.array(img).astype(np.float32)
        mask_array = np.array(mask)
        
        if mask_array.max() > 0:
            y_indices, x_indices = np.where(mask_array > 0)
            x_min, x_max = max(0, x_indices.min() - 5), min(img_array.shape[1], x_indices.max() + 5)
            y_min, y_max = max(0, y_indices.min() - 5), min(img_array.shape[0], y_indices.max() + 5)
            
            noise = np.random.normal(0., 0.03, (y_max-y_min, x_max-x_min, 3)).astype(np.float32)
            img_array[y_min:y_max, x_min:x_max] += noise * 255
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def random_erase(self, img, mask):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img)
        mask_array = np.array(mask)
        
        h, w = img_array.shape[:2]
        area = h * w
        target_area = random.uniform(0.02, 0.15) * area
        aspect_ratio = random.uniform(0.3, 1.0)
        
        h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if w_erase < w and h_erase < h:
            x1 = random.randint(0, w - w_erase)
            y1 = random.randint(0, h - h_erase)
            
            erase_region = mask_array[y1:y1+h_erase, x1:x1+w_erase] == 0
            if erase_region.any():
                img_array[y1:y1+h_erase, x1:x1+w_erase][erase_region] = random.randint(0, 255)
        
        return Image.fromarray(img_array)

    def simulate_lighting(self, img):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        effect = random.choice(['multi_spot'])
        
        if effect == 'side_light':
            angle = random.uniform(0, 2*np.pi)
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x, y)
            
            gradient = np.cos(angle) * xx/w + np.sin(angle) * yy/h
            gradient = (gradient + 1) / 2
            gradient = gradient * 1.4 + 0.3
            
            img_array *= gradient[:, :, np.newaxis]
        
        elif effect == 'multi_spot':
            n_spots = random.randint(2, 4)
            light_mask = np.ones((h, w))
            
            for _ in range(n_spots):
                center_x = random.randint(0, w)
                center_y = random.randint(0, h)
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                spot = 1.5 * np.exp(-dist / (w/5))
                light_mask += spot
            
            light_mask = (light_mask - light_mask.min()) / (light_mask.max() - light_mask.min())
            light_mask = light_mask * 1.4 + 0.3
            img_array *= light_mask[:, :, np.newaxis]
        
        elif effect == 'ring_light':
            center_x = w // 2
            center_y = h // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            ring_radius = min(w, h) // 3
            ring_width = ring_radius // 2
            ring = np.exp(-((dist - ring_radius)**2) / (2 * ring_width**2))
            ring = ring * 1.4 + 0.3
            
            img_array *= ring[:, :, np.newaxis]
        
        elif effect == 'backlight':
            center_x = w // 2
            center_y = h // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            backlight = 1 - dist/max_dist
            backlight = backlight * 1.4 + 0.3
            img_array *= backlight[:, :, np.newaxis]
        
        elif effect == 'strobe':
            stripe_width = random.randint(20, 50)
            angle = random.uniform(0, np.pi)
            
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x, y)
            rotated_pos = xx * np.cos(angle) + yy * np.sin(angle)
            stripes = np.sin(2 * np.pi * rotated_pos / stripe_width)
            stripes = (stripes + 1) / 2
            stripes = stripes * 0.8 + 0.6
            
            img_array *= stripes[:, :, np.newaxis]
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def simulate_oil_mist(self, img):
        if random.random() > 0.8:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        mist = np.zeros((h, w))
        
        num_spots = random.randint(3, 7)
        for _ in range(num_spots):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            
            size_x = random.randint(w//8, w//3)
            size_y = random.randint(h//8, h//3)
            
            y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            
            spot = (x*x)/(size_x*size_x) + (y*y)/(size_y*size_y) <= 1
            
            distortion = np.fromfunction(
                lambda i, j: np.sin(i/30 + random.uniform(0, 2*np.pi)) * 
                           np.cos(j/30 + random.uniform(0, 2*np.pi)), 
                (h, w)
            )
            
            spot = spot.astype(float) * (0.5 + 0.5 * distortion)
            
            mist += spot
            
        flow = np.fromfunction(
            lambda i, j: np.sin(i/50 + j/70) * np.cos(j/60 - i/40),
            (h, w)
        )
        mist += 0.3 * flow
        
        mist = (mist - mist.min()) / (mist.max() - mist.min())
        
        mist = gaussian_filter(mist, sigma=random.uniform(2, 4))
        
        refraction = np.fromfunction(
            lambda i, j: np.sin(i/100) * np.cos(j/100),
            (h, w)
        )
        refraction = gaussian_filter(refraction, sigma=2) * 3
        
        for c in range(3):
            shifted = np.roll(img_array[:,:,c], 
                            shift=int(refraction.max() * 3), 
                            axis=1)
            img_array[:,:,c] = img_array[:,:,c] * (1 - mist) + shifted * mist
        
        for c in range(3):
            blurred = gaussian_filter(img_array[:,:,c], sigma=1.5)
            img_array[:,:,c] = img_array[:,:,c] * (1 - mist) + blurred * mist
        
        mist = mist[:,:,np.newaxis] * 0.8
        img_array = img_array * (1 - mist) + (255 * mist * 0.4)
        
        for c in range(3):
            img_array[:,:,c] = gaussian_filter(img_array[:,:,c], sigma=0.5)
            
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def simulate_lens_contamination(self, img):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        dust_layer = np.ones((h, w))
        
        num_spots = random.randint(5, 15)
        for _ in range(num_spots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(2, 10)
            intensity = random.uniform(0.3, 0.9)
            
            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_grid**2 + y_grid**2 <= radius**2
            
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if mask[i+radius, j+radius]:
                        yi, xi = y+i, x+j
                        if 0 <= yi < h and 0 <= xi < w:
                            dust_layer[yi, xi] = intensity
        
        dust_layer = gaussian_filter(dust_layer, sigma=1)
        dust_layer = dust_layer[:,:,np.newaxis]
        
        img_array = img_array * dust_layer
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def simulate_motion_blur(self, img):
        if random.random() > 0.5:
            return img
            
        angle = random.uniform(0, 360)
        strength = random.randint(5, 6)
        
        kernel = np.zeros((strength, strength))
        center = strength // 2
        
        radian = np.deg2rad(angle)
        x_cos = np.cos(radian)
        y_sin = np.sin(radian)
        
        for i in range(strength):
            x = int(center + (i - center) * x_cos)
            y = int(center + (i - center) * y_sin)
            if 0 <= x < strength and 0 <= y < strength:
                kernel[y, x] = 1
        
        kernel = kernel / kernel.sum()
        
        img_array = np.array(img)
        channels = []
        for i in range(3):
            channels.append(cv2.filter2D(img_array[:,:,i], -1, kernel))
        blurred = np.stack(channels, axis=2)
        
        return Image.fromarray(blurred.astype(np.uint8))

    def simulate_shadow(self, img):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        shadow_type = random.choice(['gradient', 'spot', 'pattern'])
        
        if shadow_type == 'gradient':
            angle = random.uniform(0, 2*np.pi)
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x, y)
            shadow = np.cos(angle) * xx/w + np.sin(angle) * yy/h
            shadow = 0.7 + 0.3 * shadow
            
        elif shadow_type == 'spot':
            shadow = np.ones((h, w))
            num_shadows = random.randint(1, 3)
            for _ in range(num_shadows):
                center_x = random.randint(0, w)
                center_y = random.randint(0, h)
                radius = random.randint(w//8, w//4)
                y_grid, x_grid = np.ogrid[-h:h, -w:w]
                dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                shadow *= np.clip(dist/(2*radius), 0.7, 1.0)[:h,:w]
                
        else:
            freq = random.uniform(2, 5)
            angle = random.uniform(0, np.pi)
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x, y)
            rotated_pos = xx * np.cos(angle) + yy * np.sin(angle)
            shadow = 0.8 + 0.2 * np.sin(2 * np.pi * rotated_pos / (w/freq))
        
        shadow = shadow[:,:,np.newaxis]
        img_array *= shadow
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def simulate_defocus(self, img):
        if random.random() > 0.3:
            return img
        from PIL import Image, ImageFilter
        blur_radius = random.uniform(1.0, 3.0)
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def simulate_vibration(self, img):
        if random.random() > 0.5:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        displacement = np.fromfunction(
            lambda i, j: np.sin(i/10) * 2 + np.cos(j/8) * 2,
            (h, w)
        )
        
        result = np.zeros_like(img_array)
        for i in range(h):
            shift = int(displacement[i, 0])
            result[i] = np.roll(img_array[i], shift, axis=0)
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def simulate_uneven_exposure(self, img):
        if random.random() > 0.3:
            return img
            
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        exposure_type = random.choice(['vignette', 'gradient', 'random'])
        
        if exposure_type == 'vignette':
            center_x, center_y = w/2, h/2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            exposure = 1 - 0.3 * (dist / np.max(dist))
            
        elif exposure_type == 'gradient':
            angle = random.uniform(0, 2*np.pi)
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x, y)
            exposure = 0.7 + 0.6 * (np.cos(angle) * xx/w + np.sin(angle) * yy/h)
            
        else:
            exposure = np.ones((h, w))
            num_regions = random.randint(3, 6)
            for _ in range(num_regions):
                center_x = random.randint(0, w)
                center_y = random.randint(0, h)
                radius = random.randint(w//6, w//3)
                y_grid, x_grid = np.ogrid[-h:h, -w:w]
                dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                region = np.exp(-dist**2/(2*radius**2))[:h,:w]
                exposure += 0.3 * region
        
        exposure = (exposure - exposure.min()) / (exposure.max() - exposure.min())
        exposure = 0.7 + 0.6 * exposure
        exposure = exposure[:,:,np.newaxis]
        
        img_array *= exposure
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_info[image_id]
        annotations = self.image_annotations[image_id]
        
        if not annotations:
            raise IndexError(f"No annotations found for image {image_id}")
            
        ann = annotations[0]
        category_id = ann['category_id']
        category_name = self.categories[category_id]
        
        image_path = self.image_dir / category_name / img_info['file_name']
        image = load_image_with_chinese_path(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        orig_w, orig_h = image.size
        
        mask = None
        if 'segmentation' in ann:
            mask = self.create_binary_mask(ann['segmentation'], orig_h, orig_w)
            mask = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
        else:
            mask = Image.new('L', (orig_w, orig_h), 0)
        
        if 'bbox' in ann:
            bbox = ann['bbox']
        else:
            bbox = [0, 0, orig_w, orig_h]
        
        orig_image = image.copy()
        orig_mask = mask.copy()
        
        if self.split == 'train':
            if self.train_augmentation is not None:
                image = self.train_augmentation(image)
                image = self.simulate_lens_contamination(image)
        
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)
        
        bbox = self.convert_bbox_format(bbox, orig_w, orig_h)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        image = self.normalize(image)
        mask = transforms.ToTensor()(mask)
        
        label = self.category_to_onehot[category_id]
        
        description = ann.get('processed_description', None)
        if description is None and not self.test_mode:
            description = generate_defect_description(
                np.array(image),
                np.array(mask),
                defect_type=category_name
            )
            if description.startswith("Unable"):
                description = f"A defect in the fabric"
        
        return {
            'image': image,
            'bbox': bbox,
            'label': label,
            'mask': mask,
            'image_id': img_info['file_name'],
            'category_id': category_id,
            'description': description,
            'img_shape': (224, 224),
            'ori_shape': (orig_h, orig_w)
        }

def create_dataloaders(train_dir, val_dir, train_annotation_file, val_annotation_file, batch_size=8, num_workers=4, few_shot_config=None):
    logger = logging.getLogger(__name__)
    
    train_dataset = DefectDataset(
        image_dir=train_dir,
        annotation_file=train_annotation_file,
        split='train',
        test_mode=False,
        few_shot_config=few_shot_config
    )
    
    val_dataset = DefectDataset(
        image_dir=val_dir,
        annotation_file=val_annotation_file,
        split='val',
        test_mode=False
    )
    
    train_class_dist = {}
    val_class_dist = {}
    
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        category_id = sample['category_id']
        train_class_dist[category_id] = train_class_dist.get(category_id, 0) + 1
    
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        category_id = sample['category_id']
        val_class_dist[category_id] = val_class_dist.get(category_id, 0) + 1
    
    logger.info("\n类别分布统计:")
    logger.info("训练集:")
    for cat_id, count in sorted(train_class_dist.items()):
        cat_name = train_dataset.categories[cat_id]
        logger.info(f"类别 {cat_id} ({cat_name}): {count} 个样本")
    
    logger.info("\n验证集:")
    for cat_id, count in sorted(val_class_dist.items()):
        cat_name = val_dataset.categories[cat_id]
        logger.info(f"类别 {cat_id} ({cat_name}): {count} 个样本")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader
