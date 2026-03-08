# Save this as train_x3d.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

# --- Step 1: Custom Video Dataset Class for X3D ---
class VideoActionDatasetX3D(Dataset):
    """
    This Dataset is specifically designed for the X3D model.
    It loads a video clip defined by a start and end frame from a CSV,
    finds the largest person, crops the clip around them, and returns a 
    single tensor ready for the X3D model.
    """
    def __init__(self, csv_file, yolo_model, class_map, clip_len=13, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.yolo = yolo_model
        self.class_to_idx = class_map
        self.clip_len = clip_len
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            annotation = self.annotations.iloc[idx]
            video_path = annotation['video_path']
            start_frame = int(annotation['clip_start'])
            end_frame = int(annotation['clip_end'])
            activity = annotation['activity']

            if activity not in self.class_to_idx: return None
            label = self.class_to_idx[activity]

            if not os.path.exists(video_path): return None

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            if not frames: return None

            # Uniformly sample frames to match the model's expected clip length
            indices = np.linspace(0, len(frames) - 1, self.clip_len, dtype=int)
            sampled_frames = [frames[i] for i in indices]

            # Use the middle frame to find the person to crop
            middle_frame = sampled_frames[self.clip_len // 2]
            results = self.yolo(middle_frame, verbose=False, classes=[0])
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                # Find the largest person
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                x1, y1, x2, y2 = boxes[np.argmax(areas)].astype(int)
            else:
                # If no person is found, use the full frame
                h, w, _ = middle_frame.shape
                x1, y1, x2, y2 = 0, 0, w, h

            # Crop all frames in the clip using the same bounding box
            cropped_clip = [frame[y1:y2, x1:x2] for frame in sampled_frames]

            # Apply transformations
            if self.transform:
                transformed_frames = [self.transform(img) for img in cropped_clip]
            
            # Stack frames into a single tensor for X3D
            video_tensor = torch.stack(transformed_frames, dim=1)
            
            return video_tensor, label
        except Exception as e:
            # Return None if any error occurs (e.g., corrupted video)
            print(f"Skipping video due to error: {e}")
            return None

def collate_fn(batch):
    # Filters out None values from the batch, which occur if a video is corrupt
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Step 2: Evaluation Function ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            if inputs is None or labels is None: continue
            
            # X3D expects a single tensor input
            videos, labels = inputs.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct_predictions / total_samples if total_samples > 0 else 0
    model.train()
    return avg_loss, accuracy

# --- Step 3: The Main Training and Validation Function ---
def main():
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_ANNOTATIONS = "annotations/train.csv"
    VAL_ANNOTATIONS = "annotations/val.csv"
    CHECKPOINT_PATH = "x3d_training_checkpoint.pth"
    BEST_MODEL_PATH = "x3d_best_model.pth"

    CLASS_MAP = {
        'approaching atm': 0, 'card_out': 1, 'cash_out': 2, 'inserting card': 3,
        'interacting with atm': 4, 'leaving': 5, 'looking at atm': 6, 'no person': 7, 'suspicious': 8
    }
    NUM_CLASSES = len(CLASS_MAP)
    
    # --- Hyperparameters ---
    EPOCHS = 80 # X3D is smaller, can train for more epochs
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4 # A smaller learning rate is better for fine-tuning
    CLIP_DURATION = 13 # X3D_S uses 13 frames
    INPUT_CROP_SIZE = 160 # X3D_S uses a smaller input size

    # --- Model & Data Prep ---
    print(f"Using device: {DEVICE}")
    yolo_model = YOLO("yolov8n.pt")

    # Data Augmentation for training
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        torchvision.transforms.Resize((INPUT_CROP_SIZE, INPUT_CROP_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])
    
    # Simple transform for validation (no augmentation)
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((INPUT_CROP_SIZE, INPUT_CROP_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    # --- Create Datasets and DataLoaders ---
    print(f"Loading datasets... Your classes are:\n{CLASS_MAP}")
    train_dataset = VideoActionDatasetX3D(csv_file=TRAIN_ANNOTATIONS, yolo_model=yolo_model, class_map=CLASS_MAP, clip_len=CLIP_DURATION, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    val_dataset = VideoActionDatasetX3D(csv_file=VAL_ANNOTATIONS, yolo_model=yolo_model, class_map=CLASS_MAP, clip_len=CLIP_DURATION, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Load Model, Optimizer ---
    print("Loading pre-trained X3D-S model for fine-tuning...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    
    # Freeze all layers
    for param in model.parameters(): param.requires_grad = False
        
    # Unfreeze final two blocks
    for block_num in [4, 5]:
        for param in model.blocks[block_num].parameters(): param.requires_grad = True
            
    # Replace head for our number of classes
    num_ftrs = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizer will only update the unfrozen parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 0
    best_val_accuracy = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at '{CHECKPOINT_PATH}'. Resuming training.")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from Epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Main Loop ---
    print("\n" + "="*20 + " STARTING FINE-TUNING X3D " + "="*20)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for inputs, labels in train_loop:
            if inputs is None or labels is None: continue

            # X3D expects a single tensor input
            videos, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, DEVICE)
        
        print("-" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Average Train Loss: {running_loss / len(train_dataloader):.4f}")
        print(f"  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🎉 New best model saved with accuracy: {val_accuracy:.2f}%")
        print("-" * 60)

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'scheduler_state_dict': scheduler.state_dict()
        }, CHECKPOINT_PATH)

    print(f"✅ Finished Fine-Tuning! Best model saved to '{BEST_MODEL_PATH}'")

if __name__ == '__main__':
    # 'spawn' is often more stable for multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)
    main()
