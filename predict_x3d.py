# Final, corrected predict_x3d.py script

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from collections import deque
from ultralytics import YOLO

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = 'yolov8n.pt'
ACTION_MODEL_PATH = 'x3d_best_model.pth' # Your newly trained X3D model

CLASS_MAP = {
    'approaching atm': 0, 'card_out': 1, 'cash_out': 2, 'inserting card': 3,
    'interacting with atm': 4, 'leaving': 5, 'looking at atm': 6, 'no person': 7, 'suspicious': 8
}
ID_TO_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

# --- Parameters MUST match the training script ---
CLIP_DURATION = 13
CROP_SIZE = 160

# --- Preprocessing constants ---
MEAN = torch.tensor([0.45, 0.45, 0.45], device=DEVICE).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225], device=DEVICE).view(1, 3, 1, 1, 1)

# ==============================================================================
#                           MAIN ANALYSIS FUNCTION
# ==============================================================================
def analyze_video(video_path: str):
    print(f"Loading models on {DEVICE.upper()}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    action_model = torch.hub.load(
        'facebookresearch/pytorchvideo', 'x3d_s', pretrained=False
    )
    num_ftrs = action_model.blocks[5].proj.in_features
    action_model.blocks[5].proj = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    
    action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    action_model = action_model.to(DEVICE)
    action_model.eval()
    print("✅ Models loaded successfully.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"--- ERROR: Could not open video file: {video_path} ---")
        return

    clip_frames_buffer = deque(maxlen=CLIP_DURATION)
    latest_prediction = "Initializing..."
    confidence = 0.0
    
    frame_height, frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, verbose=False, classes=[0], conf=0.5)
        largest_person_box = None
        largest_area = 0
        
        if results[0].boxes:
            for box in results[0].boxes:
                area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                if area > largest_area:
                    largest_area = area
                    largest_person_box = box.xyxy[0].cpu().numpy().astype(int)

        frame_to_add = None
        if largest_person_box is not None:
            x1, y1, x2, y2 = largest_person_box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            box_dim = max(x2 - x1, y2 - y1)
            new_x1 = max(0, center_x - box_dim // 2)
            new_y1 = max(0, center_y - box_dim // 2)
            new_x2 = min(frame_width, new_x1 + box_dim)
            new_y2 = min(frame_height, new_y1 + box_dim)
            square_crop = frame[new_y1:new_y2, new_x1:new_x2]
            if square_crop.shape[0] > 0 and square_crop.shape[1] > 0:
                frame_to_add = cv2.resize(square_crop, (CROP_SIZE, CROP_SIZE))
        else:
            frame_to_add = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
        
        if frame_to_add is not None:
            rgb_frame = cv2.cvtColor(frame_to_add, cv2.COLOR_BGR2RGB)
            clip_frames_buffer.append(rgb_frame)

        if len(clip_frames_buffer) == CLIP_DURATION:
            video_tensor = torch.from_numpy(np.stack(list(clip_frames_buffer))).to(DEVICE)
            video_tensor = video_tensor.unsqueeze(0).permute(0, 4, 1, 2, 3).float()
            video_tensor = (video_tensor / 255.0 - MEAN) / STD

            # =================================================================
            # CORRECTED: Pass the tensor directly to the X3D model
            # =================================================================
            with torch.no_grad():
                preds = action_model(video_tensor)
            # =================================================================

            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_probs, pred_classes = preds.topk(k=1)
            
            pred_class_id = int(pred_classes[0][0])
            latest_prediction = ID_TO_CLASS_MAP.get(pred_class_id, "Unknown")
            confidence = pred_probs[0][0].item() * 100

        display_text = f"{latest_prediction} ({confidence:.1f}%)"
        if largest_person_box is not None:
            x1, y1, x2, y2 = largest_person_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person in frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Last Action: {display_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

        cv2.imshow('X3D Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo processing finished.")

# ==============================================================================
#                                 ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze a video with a fine-tuned X3D model.")
    parser.add_argument("video_path", type=str, help="Path to the video file to be analyzed.")
    args = parser.parse_args()
    
    analyze_video(args.video_path)
