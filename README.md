# Image-Restoration
Visual Recognitionusing Deep Learning HW4
> Student ID: 313551062
> Name: 李旻融

## Project Overview

## Repository Structure

```
image_restoration/
├── requirements.txt
├── README.md
└── src
    ├── __init__.py
    ├── dataset.py        # dataloader for train / val / test
    ├── model.py          # a light UNet-style CNN
    ├── train.py          # training script (from-scratch)
    ├── test.py          # generate pred.npz for submission
    ├── schedulers.py          # custom learning rate schedulers
```

## Implementation

## Getting Start

1. Install dependencies
```
pip install -r requirements.txt
```

2. Prepare Dataset
Ensure the following structure:
```
data/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png
│   │   ├── ...
│   │   ├── rain-1600.png
│   │   ├── snow-1.png
│   │   ├── ...
│   │   └── snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png
│       ├── ...
│       ├── rain_clean-1600.png
│       ├── snow_clean-1.png
│       ├── ...
│       └── snow_clean-1600.png
└── test/
    └── degraded
        ├── 0.png
        ├── 1.png
        ├── ...
        └── 99.png
```

3. Train the model
```
python train.py
```

4. Run Inference
```
python test.py --ckpt_path checkpoints/{checkpoint}
```
