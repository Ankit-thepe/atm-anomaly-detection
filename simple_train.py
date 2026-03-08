# ultimate_simple_train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import os

# Ultra Simple Model that WILL work
class UltraSimpleVideoModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Simple 3D CNN
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x shape: (batch, channels, frames, height, width)
        features = self.conv3d(x)
        features = features.view(x.size(0), -1)
        return self.classifier(features)

# Simple Dataset
class SimpleVideoDataset(Dataset):
    def __init__(self, csv_file, num_frames=16, img_size=112):
        self.data = pd.read_csv(csv_file, header=None, sep=' ')
        self.num_frames = num_frames
        self.img_size = img_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, start, end, label = self.data.iloc[idx]
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            # Return dummy data if video can't be read
            dummy_frames = np.zeros((self.num_frames, 3, self.img_size, self.img_size), dtype=np.float32)
            return torch.tensor(dummy_frames), int(label)
        
        # Sample frames
        for i in range(self.num_frames):
            frame_idx = min(int(start) + (int(end) - int(start)) * i // self.num_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Ensure 3 channels
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC to CHW
                frames.append(frame)
            else:
                frames.append(np.zeros((3, self.img_size, self.img_size), dtype=np.float32))
        
        cap.release()
        
        # Convert to tensor: (frames, channels, height, width)
        video_tensor = torch.tensor(np.array(frames))
        # Permute to: (channels, frames, height, width) for 3D CNN
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor, int(label)

def ultimate_train():
    print("🚀 ULTIMATE SIMPLE VIDEO TRAINING")
    print("="*50)
    print("This WILL work - no complex dependencies!")
    print("="*50)
    
    # Create model
    model = UltraSimpleVideoModel(num_classes=8)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = SimpleVideoDataset('/home/mitsdu/atm_experiment/annotations/train.csv')
    val_dataset = SimpleVideoDataset('/home/mitsdu/atm_experiment/annotations/val.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"✅ Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs('/home/mitsdu/atm_experiment/models/ultimate_simple', exist_ok=True)
    
    # Training loop
    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'🏋️ Training Epoch {epoch+1}/10')
        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"✅ Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.1f}%")
        print(f"   Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.1f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader)
        }
        
        torch.save(checkpoint, f'/home/mitsdu/atm_experiment/models/ultimate_simple/epoch_{epoch+1}.pth')
        print(f"💾 Saved: ultimate_simple/epoch_{epoch+1}.pth")
    
    print("🎉 ULTIMATE TRAINING COMPLETED!")
    print("📁 Models saved in: /home/mitsdu/atm_experiment/models/ultimate_simple/")

if __name__ == "__main__":
    ultimate_train()
