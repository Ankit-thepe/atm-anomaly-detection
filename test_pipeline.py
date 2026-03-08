# test_pipeline.py
import pandas as pd
import os

print("🧪 Testing ATM Training Pipeline")
print("="*50)

# 1. Check labels exist
labels_path = "/home/mitsdu/atm_experiment/annotations/labels_cleaned.csv"
if os.path.exists(labels_path):
    df = pd.read_csv(labels_path)
    print(f"✅ Labels file: {len(df)} clips, {df['activity'].nunique()} activities")
else:
    print("❌ Labels file not found")

# 2. Check videos exist
video_paths = df['video_path'].unique()
for vp in video_paths:
    if os.path.exists(vp):
        print(f"✅ Video exists: {os.path.basename(vp)}")
    else:
        print(f"❌ Video missing: {vp}")

# 3. Check SlowFast environment
try:
    import slowfast
    from slowfast.config.defaults import get_cfg
    print("✅ SlowFast imports correctly")
except ImportError as e:
    print(f"❌ SlowFast import error: {e}")

print("="*50)
print("If all checks pass, you're ready to train!")
