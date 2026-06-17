import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from ProDAS import ProDAS
from metric import SegmentationMetric
from dataloder.dataloder import create_dataloaders
from losses import MultiModalSegLoss  
from dataloder.crack500_dataloder import create_dataloaders as create_crack500_dataloaders
from dataloder.keloktordataloder import create_dataloaders as create_kolektor_dataloaders
from dataloder.gapsdataloder import create_dataloaders as create_gaps_dataloaders

class ProDASTrainer:
    def __init__(self,
                 model_cfg,
                 data_cfg,
                 work_dir,
                 batch_size=4,
                 num_workers=4,
                 lr=0.001,
                 epochs=100,
                 eval_interval=1,
                 save_interval=50,
                 aux_weight=0.4,
                 dataset_type='fabric'):
        
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
        self.dataset_type = dataset_type
        
        self.model = ProDAS(**model_cfg)
        self.model = self.model.cuda()
        
        if dataset_type == 'fabric':
            self.train_loader, self.val_loader = create_dataloaders(
                train_dir=data_cfg['train_dir'],
                val_dir=data_cfg['val_dir'],
                train_annotation_file=data_cfg['train_annotation_file'],
                val_annotation_file=data_cfg['val_annotation_file'],
                batch_size=batch_size,
                num_workers=num_workers
            )
        elif dataset_type == 'crack500':
            self.train_loader, self.val_loader = create_crack500_dataloaders(
                data_dir=data_cfg['data_dir'],
                batch_size=batch_size,
                num_workers=num_workers,
                debug=data_cfg.get('debug', False)
            )
        elif dataset_type == 'kolektor':
            self.train_loader, self.val_loader = create_kolektor_dataloaders(
                data_dir=data_cfg['data_dir'],
                batch_size=batch_size,
                num_workers=num_workers,
                debug=data_cfg.get('debug', False)
            )
        elif dataset_type == 'gaps':
            self.train_loader, self.val_loader = create_gaps_dataloaders(
                data_dir=data_cfg['data_dir'],
                batch_size=batch_size,
                num_workers=num_workers,
                debug=data_cfg.get('debug', False)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        self.criterion = MultiModalSegLoss(bce_weight=1.0, dice_weight=2.0, focal_weight=0.5)
        self.aux_weight = aux_weight
        
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        self.metric = SegmentationMetric(num_classes=2)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_pos_loss = 0
        total_aux_loss = 0
        valid_batches = 0
        
        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}') as pbar:
            for batch_idx, data in enumerate(self.train_loader):
                if data is None:
                    pbar.update(1)
                    continue
                
                try:
                    imgs = data['image'].cuda()
                    masks = data['mask'].cuda()
                    descriptions = data.get('description', None)
                    masks = (masks > 0.5).float()
                    
                    self.optimizer.zero_grad()
                    output, aux_outputs = self.model(imgs, descriptions)
                    
                    loss_dict = self.criterion(output, masks)
                    seg_loss = loss_dict['total_loss']
                    
                    aux_loss = 0
                    if aux_outputs:
                        for aux_out in aux_outputs:
                            if aux_out.shape[2:] != masks.shape[2:]:
                                aux_out = F.interpolate(aux_out, size=masks.shape[2:], mode='bilinear', align_corners=False)
                            aux_loss_dict = self.criterion(aux_out, masks)
                            aux_loss += aux_loss_dict['total_loss']
                        aux_loss = aux_loss / len(aux_outputs)
                    
                    total_loss = seg_loss + self.aux_weight * aux_loss
                    
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    self.optimizer.step()
                    
                    batch_total_loss = total_loss.item()
                    total_aux_loss += aux_loss.item()
                    
                    valid_batches += 1
                    avg_loss = batch_total_loss
                    avg_aux_loss = total_aux_loss / valid_batches
                    
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'bce': f'{loss_dict["bce_loss"]:.4f}',
                        'dice': f'{loss_dict["dice_loss"]:.4f}',
                        'focal': f'{loss_dict["focal_loss"]:.4f}',
                        'aux': f'{avg_aux_loss:.4f}'
                    })
                
                except Exception as e:
                    continue
                
                pbar.update(1)
        
        return avg_loss if valid_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.metric.reset()
        
        class_metrics = {}
        id_to_name = {v: k for k, v in self.val_loader.dataset.categories.items()}
        for class_id in range(self.val_loader.dataset.num_classes):
            class_metrics[class_id] = SegmentationMetric(num_classes=2)
        
        for data in tqdm(self.val_loader, desc='Evaluating'):
            imgs = data['image'].cuda()
            masks = data['mask'].cuda()
            labels = data['label']
            descriptions = data.get('description', None)
            masks = (masks > 0.5).long()
            batch_size = imgs.shape[0]
            
            pred, _ = self.model(imgs, descriptions)  
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).long()
            
            self.metric.update(pred, masks)
            
            for i in range(batch_size):
                class_id = labels[i].argmax().item()
                class_metrics[class_id].update(pred[i:i+1], masks[i:i+1])
        
        eval_results = {
            'mIoU': self.metric.mean_iou(),
            'Dice': self.metric.dice_coefficient(),
            'Pixel_Acc': self.metric.pixel_accuracy()
        }
        
        for class_id, metric in class_metrics.items():
            if class_id in id_to_name:
                category_name = id_to_name[class_id]
                eval_results[f'IoU_{category_name}'] = metric.mean_iou()
                eval_results[f'Dice_{category_name}'] = metric.dice_coefficient()
            else:
                eval_results[f'IoU_class_{class_id}'] = metric.mean_iou()
                eval_results[f'Dice_class_{class_id}'] = metric.dice_coefficient()
        
        return eval_results 

    def save_checkpoint(self, epoch, eval_results=None):
        state_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if eval_results is not None:
            state_dict['eval_results'] = eval_results
            
        save_path = os.path.join(self.work_dir, f'epoch_{epoch}.pth')
        torch.save(state_dict, save_path)

    def train(self):
        best_miou = 0
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            
            if epoch % self.eval_interval == 0:
                eval_results = self.evaluate()
                
                if eval_results['mIoU'] > best_miou:
                    best_miou = eval_results['mIoU']
                    self.save_checkpoint(epoch, eval_results)
            
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)

if __name__ == '__main__':
    dataset_type = 'isic'
    
    if dataset_type == 'fabric':
        data_cfg = {
            'train_dir': 'dataset/训练',
            'val_dir': 'dataset/测试',
            'train_annotation_file': 'processed_annotations/train_annotations_coco.json',
            'val_annotation_file': 'processed_annotations/test_annotations_coco.json'
        }
    elif dataset_type == 'crack500':
        data_cfg = {
            'data_dir': 'crack500',
            'debug': True
        }
    elif dataset_type == 'kolektor':
        data_cfg = {
            'data_dir': 'kolektor',
            'debug': True
        }
    elif dataset_type == 'gaps':
        data_cfg = {
            'data_dir': 'Gaps',
            'debug': True
        }
    elif dataset_type == 'isic':
        data_cfg = {
            'data_dir': 'ISIC2018',
            'debug': True
        }
    elif dataset_type == 'mvtec':
        data_cfg = {
            'data_dir': 'mvtec',
            'debug': True
        }
    elif dataset_type == 'aitex':
        data_cfg = {
            'data_dir': 'AITEX',
            'debug': True
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    model_cfg = {
        'num_classes': 1,
        'num_channels': 3,
        'feature_scale': 2,
        'dropout': 0.2,
        'fuse': True,
        'out_ave': True
    }
    
    trainer = ProDASTrainer(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        work_dir=f'./work_dirs/ProDAS_{dataset_type}all',
        batch_size=16,
        num_workers=8,
        lr=0.001,
        epochs=100,
        dataset_type=dataset_type
    )
    
    trainer.train() 