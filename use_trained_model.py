#!/usr/bin/env python3
import torch
import os
from slowfast.models import build_model
from slowfast.config.defaults import get_cfg

class ATMModelLoader:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.cfg = None
        
    def load_model(self):
        """Load the trained ATM model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False
            
        # Load config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.config_path)
        
        # Build model
        self.model = build_model(self.cfg)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print(f"✅ Model loaded successfully: {os.path.basename(self.model_path)}")
        return True
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "Model not loaded"
        
        info = f"""
🏦 ATM Activity Recognition Model
────────────────────────────────
• Model: {self.cfg.MODEL.MODEL_NAME if self.cfg else 'Unknown'}
• Classes: {self.model.num_classes if self.model else 'Unknown'}
• Input frames: {self.cfg.DATA.NUM_FRAMES if self.cfg else 'Unknown'}
• Model path: {os.path.basename(self.model_path)}
        """
        return info

def list_saved_models():
    """List all saved models"""
    model_dir = "/home/mitsdu/atm_experiment/models/first_model/final_models"
    if not os.path.exists(model_dir):
        print("❌ Models directory not found")
        return
        
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not models:
        print("❌ No trained models found")
        return
        
    print("🏆 Available Trained Models:")
    print("="*50)
    for i, model_file in enumerate(sorted(models)):
        model_path = os.path.join(model_dir, model_file)
        size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f"{i+1}. {model_file} ({size:.1f} MB)")
    
    print("\nUsage: python use_trained_model.py <model_number>")

if __name__ == "__main__":
    list_saved_models()
    
    # Example of loading a model
    model_dir = "/home/mitsdu/atm_experiment/models/first_model/final_models"
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')] if os.path.exists(model_dir) else []
    
    if models:
        sample_model = os.path.join(model_dir, models[0])
        loader = ATMModelLoader(sample_model, "/home/mitsdu/atm_experiment/atm_complete_config.yaml")
        if loader.load_model():
            print(loader.get_model_info())
