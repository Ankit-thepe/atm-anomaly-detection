# Save this file as test_yolo.py

from ultralytics import YOLO
import cv2

def run_test():
    # --- CONFIGURATION ---
    # We'll use the stronger 'yolov8s.pt' model for this test
    MODEL_NAME = 'yolov8n.pt' 
    VIDEO_PATH = '10.mp4'       # The video you are testing
    CONFIDENCE = 0.10           # Use a very low 10% confidence threshold

    print(f"Loading stronger model '{MODEL_NAME}' for a quick test...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    print(f"Reading the first frame from '{VIDEO_PATH}'...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'")
        return
        
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read a frame from the video.")
        return

    # Run the prediction
    print(f"Running detection with a low {CONFIDENCE*100}% confidence threshold...")
    results = model(frame, conf=CONFIDENCE)

    # --- SHOW RESULTS ---
    print("\n--- DETECTION RESULTS ---")
    boxes = results[0].boxes
    print(f"Found {len(boxes)} objects in total.")
    for box in boxes:
        class_id = int(box.cls)
        class_name = model.model.names[class_id]
        conf = float(box.conf)
        print(f" -> Found '{class_name}' with {conf:.2f} confidence.")

    print("\nDisplaying image with detected boxes. Press any key to close.")
    results[0].show() # This will open a window showing the image with boxes
    print("Test finished.")

if __name__ == "__main__":
    run_test()
