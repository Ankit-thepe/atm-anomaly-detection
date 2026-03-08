# Save this as train_slowfast.py

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
from torch.optim.lr_scheduler import StepLR # UPDATED: Import the scheduler

# --- Step 1: Import Your Local SlowFast Model ---
try:
    from slowfast_model import SlowFast
except ImportError:
    print("="*50)
    print("ERROR: Could not import 'SlowFast' from 'slowfast_model.py'.")
    print("Make sure 'slowfast_model.py' is in the same directory.")
    exit()

# --- Step 2: Custom Video Dataset Class (CSV Version) ---
class VideoActionDatasetCSV(Dataset):
    # This class is unchanged
    def __init__(self, csv_file, yolo_model, class_map, model_clip_len=32, alpha=4, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.yolo = yolo_model
        self.class_to_idx = class_map
        self.model_clip_len = model_clip_len
        self.alpha = alpha
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
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
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: return None
        indices = np.linspace(0, len(frames) - 1, self.model_clip_len, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        middle_frame = sampled_frames[self.model_clip_len // 2]
        results = self.yolo(middle_frame, verbose=False, classes=[0])
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            x1, y1, x2, y2 = boxes[np.argmax(areas)].astype(int)
        else:
            h, w, _ = middle_frame.shape
            x1, y1, x2, y2 = 0, 0, w, h
        cropped_clip = [frame[y1:y2, x1:x2] for frame in sampled_frames]
        fast_path_frames = cropped_clip
        slow_path_indices = np.linspace(0, self.model_clip_len - 1, self.model_clip_len // self.alpha, dtype=int)
        slow_path_frames = [cropped_clip[i] for i in slow_path_indices]
        if self.transform:
            slow_path_frames = [self.transform(img) for img in slow_path_frames]
            fast_path_frames = [self.transform(img) for img in fast_path_frames]
        slow_tensor = torch.stack(slow_path_frames, dim=1)
        fast_tensor = torch.stack(fast_path_frames, dim=1)
        return [slow_tensor, fast_tensor], label

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- Step 3: Evaluation Function ---
def evaluate(model, dataloader, criterion, device):
    # This function is unchanged
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            if inputs is None or labels is None: continue
            slow_path, fast_path, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
            outputs = model([slow_path, fast_path])
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct_predictions / total_samples if total_samples > 0 else 0
    model.train()
    return avg_loss, accuracy

# --- Step 4: The Main Training and Validation Function ---
def main():
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_ANNOTATIONS = "annotations/train.csv"
    VAL_ANNOTATIONS = "annotations/val.csv"
    CHECKPOINT_PATH = "training_checkpoint.pth"
    BEST_MODEL_PATH = "slowfast_best_model.pth"

    CLASS_MAP = {
        'approaching atm': 0, 'card_out': 1, 'cash_out': 2, 'inserting card': 3,
        'interacting with atm': 4, 'leaving': 5, 'looking at atm': 6, 'no person':7, 'suspicious': 8
    }
    NUM_CLASSES = len(CLASS_MAP)
    
    # --- Hyperparameters ---
    EPOCHS = 120
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001 # Lowered LR for more stable fine-tuning

    # --- Model & Data Prep ---
    print(f"Using device: {DEVICE}")
    yolo_model = YOLO("yolov8n.pt")

    # UPDATED: Added Data Augmentation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    # --- Create Datasets and DataLoaders ---
    print(f"Loading datasets... Your classes are:\n{CLASS_MAP}")
    train_dataset = VideoActionDatasetCSV(csv_file=TRAIN_ANNOTATIONS, yolo_model=yolo_model, class_map=CLASS_MAP, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataset = VideoActionDatasetCSV(csv_file=VAL_ANNOTATIONS, yolo_model=yolo_model, class_map=CLASS_MAP, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Load Model, Optimizer ---
    print("Loading pre-trained SlowFast model for fine-tuning...")
    slowfast_model = SlowFast(num_classes=NUM_CLASSES).to(DEVICE)
    
    # UPDATED: Optimizer with differential learning rates
    optimizer = optim.Adam([
        {"params": slowfast_model.blocks[5].parameters(), "lr": LEARNING_RATE / 10},
        {"params": slowfast_model.blocks[6].parameters(), "lr": LEARNING_RATE}
    ], lr=LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss()
    
    # UPDATED: Added a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    start_epoch = 0
    best_val_accuracy = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at '{CHECKPOINT_PATH}'. Resuming training.")
        checkpoint = torch.load(CHECKPOINT_PATH)
        slowfast_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Also load scheduler state
        print(f"Resuming from Epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Main Loop ---
    print("\n" + "="*20 + " STARTING FINE-TUNING " + "="*20)
    for epoch in range(start_epoch, EPOCHS):
        slowfast_model.train()
        running_loss = 0.0
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for inputs, labels in train_loop:
            if inputs is None or labels is None: continue
            slow_path, fast_path, labels = inputs[0].to(DEVICE), inputs[1].to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = slowfast_model([slow_path, fast_path])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        val_loss, val_accuracy = evaluate(slowfast_model, val_dataloader, criterion, DEVICE)
        
        print("-" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Average Train Loss: {running_loss / len(train_dataloader):.4f}")
        print(f"  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(slowfast_model.state_dict(), BEST_MODEL_PATH)
            print(f"🎉 New best model saved with accuracy: {val_accuracy:.2f}%")
        print("-" * 60)

        # UPDATED: Step the scheduler after each epoch
        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': slowfast_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'scheduler_state_dict': scheduler.state_dict() # Also save scheduler state
        }, CHECKPOINT_PATH)

    print(f"✅ Finished Fine-Tuning! Best model saved to '{BEST_MODEL_PATH}'")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 
    main()
