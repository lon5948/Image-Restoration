# Image-Restoration
Visual Recognitionusing Deep Learning HW4
> Student ID: 313551062  
> Name: 李旻融

## Project Overview
This project tackles the image restoration problem of removing rain and snow degradations from images. Using a unified model trained from scratch on synthetically generated datasets, we adapt and enhance the PromptIR transformer-based architecture to deliver high-performance restoration results.

Key modifications include:
- Removal of decoder module
- Loosened latent bottleneck
- Deeper transformer and refinement blocks
- Test-Time Augmentation (TTA)

The model achieves 30.83 dB PSNR on the final test set.

## Repository Structure

```
image_restoration/
├── requirements.txt        # Dependency list
├── README.md               # Project overview
└── src/
    ├── __init__.py
    ├── dataset.py          # Dataset loader
    ├── model.py            # Modified PromptIR-based CNN model
    ├── schedulers.py       # Custom learning rate schedulers
    ├── train.py            # Training script
    └── test.py             # Inference script for generating predictions
```

## Dataset Structure

- 600 rain and 1600 snow images for training
- 100 test images (50 per type) with unknown degradation labels

```
data/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png
│   │   ├── ...
│   │   └── snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png
│       ├── ...
│       └── snow_clean-1600.png
└── test/
    └── degraded/
        ├── 0.png
        ├── ...
        └── 99.png
```

## Getting Start

1. Install dependencies
```
pip install -r requirements.txt
```

2. Prepare Dataset
Ensure the dataset follows the directory structure shown above.

3. Train the model
```
python train.py
```

4. Run Inference
```
python test.py --ckpt_path checkpoints/{your_checkpoint}.ckpt
```

This will generate a pred.npz file for submission.

## Model Design Summary
- Backbone Depth: [6, 8, 8, 10]
- Latent Bottleneck: 384 → 256 → 192
- Prompt Blocks: After Transformer Blocks L2 & L3
- Refinement Blocks: 6
- Decoder: Removed
- Augmentation: TTA only during inference

## Experimental Results
| Configuration                            | PSNR (dB) |
| ---------------------------------------- | --------- |
| Baseline PromptIR                        | 28.64     |
| + Decoder Removal                        | 30.09     |
| + Channel Loosening & Deeper Transformer | 30.27     |
| + TTA                                    | **30.83** |

#### Performance snapshot
![Performance Snapshot](images/snapshot.png)
