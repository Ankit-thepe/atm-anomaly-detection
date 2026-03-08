# final_fix.py
import os
import subprocess
import shutil

def use_proper_config_merging():
    """Use SlowFast's config system correctly"""
    
    print("🔧 Using Proper Config Merging")
    print("="*50)
    
    # Step 1: Copy the working Kinetics config
    base_config = "/home/mitsdu/detectron2/slowfast/configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml"
    target_config = "/home/mitsdu/atm_experiment/atm_final_config.yaml"
    
    shutil.copy(base_config, target_config)
    print("✅ Copied base Kinetics config")
    
    # Step 2: Create a simple training script that uses config merging correctly
    train_script = """#!/bin/bash
cd /home/mitsdu/detectron2/slowfast

echo "🚀 Final Training Attempt with Proper Config"
echo "============================================"

# Use the base config and override specific parameters
python tools/run_net.py \\
  --cfg configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml \\
  NUM_GPUS 1 \\
  MODEL.NUM_CLASSES 8 \\
  DATA.PATH_TO_DATA_DIR "/home/mitsdu/atm_experiment/annotations" \\
  DATA.PATH_PREFIX "/home/mitsdu/atm_experiment/videos" \\
  TRAIN.BATCH_SIZE 2 \\
  SOLVER.BASE_LR 0.0001 \\
  SOLVER.MAX_EPOCH 10 \\
  TRAIN.CHECKPOINT_FILE_PATH "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl" \\
  TRAIN.FINETUNE True \\
  OUTPUT_DIR "/home/mitsdu/atm_experiment/models/first_model" \\
  2>&1 | tee /home/mitsdu/atm_experiment/final_training.log
"""
    
    with open("/home/mitsdu/atm_experiment/final_train.sh", "w") as f:
        f.write(train_script)
    
    os.chmod("/home/mitsdu/atm_experiment/final_train.sh", 0o755)
    print("✅ Created final training script")

def try_minimal_override():
    """Try minimal parameter override approach"""
    
    print("\\n🎯 Attempting Minimal Parameter Override")
    
    cmd = [
        'python', 'tools/run_net.py',
        '--cfg', 'configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml',
        'NUM_GPUS', '1',
        'MODEL.NUM_CLASSES', '8',
        'DATA.PATH_TO_DATA_DIR', '/home/mitsdu/atm_experiment/annotations',
        'OUTPUT_DIR', '/home/mitsdu/atm_experiment/models/first_model'
    ]
    
    print("Command:", ' '.join(cmd))
    
    os.chdir('/home/mitsdu/detectron2/slowfast')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        print("Return code:", result.returncode)
    except Exception as e:
        print(f"Error: {e}")

def check_config_system():
    """Check how SlowFast's config system works"""
    
    print("\\n🔍 Checking SlowFast Config System")
    
    test_script = """
import sys
sys.path.append('/home/mitsdu/detectron2/slowfast')

from slowfast.config.defaults import get_cfg

# Test 1: Basic config
cfg = get_cfg()
print("✅ Basic config created")

# Test 2: Load Kinetics config
cfg.merge_from_file('/home/mitsdu/detectron2/slowfast/configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml')
print("✅ Kinetics config loaded")
print(f"Model classes: {cfg.MODEL.NUM_CLASSES}")

# Test 3: Override parameters
cfg.MODEL.NUM_CLASSES = 8
cfg.DATA.PATH_TO_DATA_DIR = '/home/mitsdu/atm_experiment/annotations'
print("✅ Parameters overridden")
print(f"New model classes: {cfg.MODEL.NUM_CLASSES}")

# Test 4: Try to build model
from slowfast.models import build_model
try:
    model = build_model(cfg)
    print("✅ Model built successfully!")
except Exception as e:
    print(f"❌ Model build failed: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open("/home/mitsdu/atm_experiment/test_config.py", "w") as f:
        f.write(test_script)
    
    os.system("cd /home/mitsdu/detectron2/slowfast && python /home/mitsdu/atm_experiment/test_config.py")

def alternative_approach_slowonly():
    """Try SlowOnly model as alternative"""
    
    print("\\n🔄 Trying SlowOnly Model Alternative")
    
    cmd = [
        'python', 'tools/run_net.py',
        '--cfg', 'configs/Kinetics/SLOWONLY_8x8_R50.yaml',
        'NUM_GPUS', '1',
        'MODEL.NUM_CLASSES', '8',
        'DATA.PATH_TO_DATA_DIR', '/home/mitsdu/atm_experiment/annotations',
        'DATA.PATH_PREFIX', '/home/mitsdu/atm_experiment/videos',
        'TRAIN.BATCH_SIZE', '2',
        'SOLVER.BASE_LR', '0.0001',
        'SOLVER.MAX_EPOCH', '5',
        'OUTPUT_DIR', '/home/mitsdu/atm_experiment/models/slowonly_model'
    ]
    
    print("SlowOnly command:", ' '.join(cmd))
    
    os.chdir('/home/mitsdu/detectron2/slowfast')
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SlowOnly training started successfully!")
    else:
        print("❌ SlowAlso failed")
        print("Error:", result.stderr[-500:] if result.stderr else "No error output")

if __name__ == "__main__":
    use_proper_config_merging()
    check_config_system()
    
    print("\\n🚀 Now trying the final training approach...")
    try_minimal_override()
    
    # If SlowFast still fails, try SlowOnly
    print("\\n🔄 If SlowFast fails, trying SlowOnly...")
    alternative_approach_slowonly()
