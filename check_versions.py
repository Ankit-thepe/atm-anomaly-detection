# check_versions.py
import torch
import torchvision
import subprocess

print("🔍 Checking Software Versions")
print("="*40)

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")

# Check SlowFast version
try:
    result = subprocess.run(['git', '-C', '/home/mitsdu/detectron2/slowfast', 'log', '-1', '--oneline'], 
                          capture_output=True, text=True)
    print(f"SlowFast commit: {result.stdout.strip()}")
except:
    print("SlowFast version: Unknown")

# Check if there are known issues
print("\n💡 Known issue: SlowFast may have compatibility issues with newer PyTorch versions")
print("💡 Solution: Try using an older PyTorch version or a different SlowFast branch")
