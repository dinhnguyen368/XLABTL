# Improving Edge Detection in Real-World Noisy Images Using Noise-Proxy-Adaptive Denoising

## 1. Overview

This project studies whether denoising preprocessing can improve Canny edge detection
on real-world images captured under different noise conditions.

Baseline:
Grayscale -> Canny

Proposed:
Grayscale -> Bilateral Filtering -> Canny

The adaptive signal is a noise proxy estimated from a homogeneous ROI.
It is not treated as a pure sensor-noise measurement.

## 2. Dataset

The dataset contains:

- 10 scenes
- 3 noise levels per scene
- 30 images total
- 5 development scenes
- 5 evaluation scenes
- 10 manually annotated Ground Truth edge maps

Directory structure:

data/
├── development/
├── evaluation/
│   ├── light/
│   ├── medium/
│   └── heavy/
├── ground_truth/
└── metadata.csv

## 3. Environment

Python 3.11

Create virtual environment:

python -m venv .venv

Activate on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## 4. Validation

Run:

python -m pytest -q

python main.py --mode validate

## 5. Experiments

Run in this order:

python main.py --mode E1

python main.py --mode E2

python main.py --mode E3

python main.py --mode E4

## 6. Experimental Design

E1:
Demonstrates baseline degradation across noise levels.

E2:
Searches Bilateral Filter parameters using the development set only.

E3:
Compares Baseline and Proposed on the evaluation set.

E4:
Applies controlled Gaussian noise to evaluation Light images using a frozen
high-noise configuration.

## 7. Noise Proxy

noise_measure is computed from the high-frequency residual of a Gaussian-blurred
homogeneous ROI.

It is an image-level high-frequency residual proxy, not pure sensor noise.

## 8. Ground Truth

Ground Truth edge maps are manually annotated.
They are not generated using Canny.

## 9. Results

Outputs are stored in:

outputs/tables/
outputs/figures/
outputs/intermediate/
outputs/configs/

## 10. Important Limitations

- The available JPG images do not contain EXIF ISO metadata.
- The adaptive method therefore uses the measured noise proxy rather than fabricated ISO values.
- Images were standardized to a common size before evaluation.
- E4 is a controlled Gaussian stress test and is not equivalent to real camera noise.
- Some real-world scenes do not show perfectly monotonic noise-proxy values across the three capture levels.