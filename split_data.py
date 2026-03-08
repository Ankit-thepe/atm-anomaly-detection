# Save this file as split_data.py in your atm_experiment directory

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
SOURCE_FILE = "annotations/labels_cleaned.csv"
OUTPUT_DIR = "annotations"
TRAIN_FILE_PATH = os.path.join(OUTPUT_DIR, "train.csv")
VAL_FILE_PATH = os.path.join(OUTPUT_DIR, "val.csv")
TEST_SIZE = 0.20  # Use 20% of the data for validation
RANDOM_STATE = 42 # Ensures the split is the same every time you run it

def split_data():
    """
    Reads the main cleaned CSV, splits it into training and validation sets,
    and saves them to the annotations directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Check if the source file exists
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Error: Source file '{SOURCE_FILE}' not found.")
        print("Please make sure your cleaned data file is in the same directory as this script.")
        return

    # Load the dataset
    print(f"Reading data from '{SOURCE_FILE}'...")
    df = pd.read_csv(SOURCE_FILE)
    print(f"Found {len(df)} total clips.")

    # Stratified split to maintain the same class distribution in train and val sets
    # This is important if some actions are rare.
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['activity']  # Ensures class balance
    )

    # Save the new files
    train_df.to_csv(TRAIN_FILE_PATH, index=False)
    val_df.to_csv(VAL_FILE_PATH, index=False)

    print("\n✅ Data successfully split!")
    print(f"   - Training data ({len(train_df)} clips) saved to: {TRAIN_FILE_PATH}")
    print(f"   - Validation data ({len(val_df)} clips) saved to: {VAL_FILE_PATH}")

if __name__ == "__main__":
    split_data()
