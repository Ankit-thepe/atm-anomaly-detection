# ATM Activity Recognition System

A deep learning system for recognizing activities at ATM locations using video surveillance footage.

## Overview

This project implements an activity recognition system that can identify 9 different activities around ATM machines:

1. **approaching_atm** - Person approaching the ATM
2. **card_out** - Taking out/inserting card
3. **cash_out** - Dispensing/taking cash
4. **inserting_card** - Card insertion action
5. **interacting** - General interaction with ATM
6. **leaving** - Person leaving ATM area
7. **looking** - Person looking around/observing
8. **no_person** - Empty scene with no person
9. **suspicious** - Suspicious behavior detected

## Models

The system supports two state-of-the-art video action recognition models:

### SlowFast Networks
- **Model**: SlowFast R50 (8x8 frames)
- **Input**: 32 frames at 224×224 resolution
- **Strengths**: Excellent accuracy, handles temporal dynamics well
- **Best for**: High-accuracy requirements

### X3D Networks  
- **Model**: X3D-S (Small variant)
- **Input**: 13 frames at 224×224 resolution
- **Strengths**: Fast inference, smaller model size
- **Best for**: Real-time deployment, edge devices

## Key Features

- **Person Detection**: Uses YOLOv8 to detect and crop person regions
- **Transfer Learning**: Pre-trained on Kinetics-400, fine-tuned on ATM data
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Real-time Inference**: Optimized for real-time video processing
- **Modular Design**: Easy to extend with new activities or models

## Dataset

- **Total Videos**: ~700 clips
- **Classes**: 9 activity types
- **Duration**: Variable length clips (5-30 seconds)
- **Resolution**: Various resolutions, normalized to 224×224
- **Annotations**: Frame-level annotations with activity labels

## Results

### SlowFast Model Performance
- **Overall Accuracy**: 85-92%
- **Best Classes**: approaching_atm, leaving, no_person (>95%)
- **Challenging Classes**: suspicious, interacting (~75-80%)

### X3D Model Performance  
- **Overall Accuracy**: 80-87%
- **Inference Speed**: ~3x faster than SlowFast
- **Model Size**: ~60% smaller than SlowFast

## Project Structure

```
atm_experiment/
├── train_slowfast.py      # SlowFast training script
├── train_x3d.py          # X3D training script
├── predict_video.py      # Video prediction with SlowFast
├── predict_x3d.py        # Video prediction with X3D
├── slowfast_model.py     # SlowFast model definition
├── simple_train.py       # Simplified training script
├── annotations/          # Dataset annotations and labels
├── models/               # Saved model checkpoints
├── videos/              # Video data
├── results/             # Training logs and results
└── checkpoints/         # Model checkpoints during training
```

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install pytorchvideo
pip install ultralytics  # for YOLOv8
pip install opencv-python
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
```

### Setup

1. Clone this repository
2. Install dependencies
3. Download pre-trained models (Kinetics-400)
4. Prepare your video dataset
5. Update paths in training scripts

## Usage

### Training SlowFast Model

```bash
python train_slowfast.py
```

### Training X3D Model

```bash
python train_x3d.py
```

### Video Prediction

```bash
# Using SlowFast
python predict_video.py --video path/to/video.mp4 --model models/slowfast_best_model.pth

# Using X3D
python predict_x3d.py --video path/to/video.mp4 --model models/x3d_best_model.pth
```

### Simple Training (Recommended for beginners)

```bash
python simple_train.py
```

## Model Architecture

### SlowFast Networks
- **Slow Pathway**: 8 frames with 8× temporal stride
- **Fast Pathway**: 64 frames with 1× temporal stride  
- **Fusion**: Lateral connections between pathways
- **Backbone**: ResNet-50
- **Output**: 9 classes (ATM activities)

### X3D Networks
- **Architecture**: 3D CNN with depthwise separable convolutions
- **Temporal Modeling**: 13 frames with efficient 3D convolutions
- **Spatial Resolution**: 224×224
- **Efficiency**: Optimized for mobile/edge deployment

## Data Preprocessing

1. **Person Detection**: YOLOv8n detects person bounding boxes
2. **Cropping**: Videos cropped to person region
3. **Temporal Sampling**: Extract fixed number of frames
4. **Spatial Normalization**: Resize to 224×224
5. **Data Augmentation**: Random flips, crops, color jittering

## Training Strategy

### Transfer Learning
- Start with Kinetics-400 pre-trained weights
- Freeze backbone layers initially
- Fine-tune classifier and last few layers
- Gradual unfreezing for better adaptation

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing schedule
- **Loss Function**: CrossEntropy with class weights
- **Batch Size**: 8-16 (depending on GPU memory)

## Deployment

### Real-time Inference
- **Preprocessing**: YOLOv8 person detection + cropping
- **Model**: X3D for speed or SlowFast for accuracy  
- **Postprocessing**: Temporal smoothing, confidence thresholding
- **Output**: Activity labels with confidence scores

### Performance Optimization
- **Mixed Precision**: FP16 training and inference
- **Model Quantization**: INT8 for edge deployment
- **Batch Processing**: Process multiple clips simultaneously
- **Temporal Caching**: Reuse features across overlapping clips

## Future Improvements

- [ ] Add more activity classes (fraud detection, accessibility assistance)
- [ ] Implement multi-person tracking and activity recognition
- [ ] Add anomaly detection for unusual behaviors
- [ ] Optimize for real-time edge deployment
- [ ] Integrate with existing security camera systems
- [ ] Add data anonymization and privacy protection

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-activity`)
3. Commit changes (`git commit -am 'Add new activity recognition'`)
4. Push to branch (`git push origin feature/new-activity`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PyTorchVideo**: Excellent video understanding library
- **Kinetics-400**: Pre-trained model weights
- **SlowFast Networks**: Original architecture from Facebook AI Research
- **X3D Networks**: Efficient video networks from Facebook AI Research
- **YOLOv8**: Person detection by Ultralytics

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{atm-activity-recognition,
  title={ATM Activity Recognition System},
  author={Ankit Thepe},
  year={2025},
  url={https://github.com/Ankit-thepe/atm-anomaly-detection}
}
```

## Contact

For questions or collaboration opportunities, please reach out:
- Email: thepeankit321@gmail.com
- LinkedIn: [Ankit-thepe6](https://www.linkedin.com/in/Ankit-thepe?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- GitHub: [@Ankit-thepe]