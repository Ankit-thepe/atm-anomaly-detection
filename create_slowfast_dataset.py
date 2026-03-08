# create_slowfast_dataset.py
import pandas as pd
import os

def prepare_slowfast_dataset(labels_csv, output_dir):
    # Load labels
    df = pd.read_csv(labels_csv)
    
    # Create train/validation split (80/20)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['activity'])
    
    # Create class mapping
    activities = sorted(df['activity'].unique())
    class_map = {activity: i for i, activity in enumerate(activities)}
    
    print("🎯 Activity Classes for Training:")
    for activity, class_id in class_map.items():
        count = len(df[df['activity'] == activity])
        print(f"  {class_id}: {activity} ({count} clips)")
    
    # Save class mapping
    with open(f"{output_dir}/class_map.txt", 'w') as f:
        for activity, class_id in class_map.items():
            f.write(f"{class_id} {activity}\n")
    
    # Save training file (SlowFast format)
    with open(f"{output_dir}/train.csv", 'w') as f:
        for _, row in train_df.iterrows():
            f.write(f"{row['video_path']} {int(row['clip_start'])} {int(row['clip_end'])} {class_map[row['activity']]}\n")
    
    # Save validation file
    with open(f"{output_dir}/val.csv", 'w') as f:
        for _, row in val_df.iterrows():
            f.write(f"{row['video_path']} {int(row['clip_start'])} {int(row['clip_end'])} {class_map[row['activity']]}\n")
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total clips: {len(df)}")
    print(f"Training clips: {len(train_df)}")
    print(f"Validation clips: {len(val_df)}")
    print(f"Number of classes: {len(activities)}")
    
    return len(activities)

# Prepare the dataset
num_classes = prepare_slowfast_dataset(
    "/home/mitsdu/atm_experiment/annotations/labels_cleaned.csv",
    "/home/mitsdu/atm_experiment/annotations"
)
