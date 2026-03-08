import pandas as pd

try:
    # Check the training file
    train_df = pd.read_csv("annotations/train.csv")
    print("✅ Columns in train.csv:", train_df.columns.tolist())

    # Check the validation file
    val_df = pd.read_csv("annotations/val.csv")
    print("✅ Columns in val.csv:  ", val_df.columns.tolist())

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
