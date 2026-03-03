# Camera Analysis & Inspection Toolkit

A compact toolkit for:

-   Image inspection & segmentation (Streamlit GUI)
-   Camera classification (CNN & Random Forest)
-   Image registration with quantitative evaluation
-   Automated PDF reporting

Compatible with Python 3.11.

------------------------------------------------------------------------

## Installation

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 1️⃣ Streamlit Inspection Pipeline (`q1.py`)

Interactive GUI for:

-   CLAHE / Gaussian preprocessing
-   Variance-based segmentation
-   Optional morphology cleanup
-   Red/Green overlay visualization
-   Batch processing for full folders

### Run

``` bash
streamlit run q1.py
```

Green = smooth regions\
Red = feature-rich regions

------------------------------------------------------------------------

## 2️⃣ CNN Camera Classifier (`q2_cnn.py`)

Binary classifier distinguishing two physical cameras.

### Features

-   Grayscale input
-   60×60 resizing
-   75/25 train-validation split
-   AUC + classification report
-   Saved model (`tiny_camera_net.pth`)
-   Reproducible (fixed seed)

### Run

``` bash
python q2_cnn.py
```

------------------------------------------------------------------------

## 3️⃣ Classical Camera Classifier (`q3.py`)

Feature-engineered approach using:

-   FFT frequency magnitude
-   Sobel edge statistics
-   Laplacian variance
-   Intensity statistics

Uses:

-   Group-aware splitting (ID-level separation)
-   GroupKFold cross-validation
-   GridSearchCV for tuning
-   RandomForest classifier

### Run

``` bash
python q3.py
```

------------------------------------------------------------------------

## 4️⃣ Image Registration & PDF Report (`q4_register.py`)

Performs:

-   ORB feature detection
-   RANSAC homography
-   SSIM before/after comparison
-   Red/Green alignment visualization
-   Automatic A4 PDF export (first 12 pairs)

### Run

``` bash
python q4_register.py
```

Output:

-   Console SSIM metrics
-   `registered_results.pdf`

------------------------------------------------------------------------

## Project Structure

    q1.py               # Streamlit inspection GUI
    q2_cnn.py           # CNN camera classifier
    q3.py               # Classical RF classifier
    q4_register.py      # Registration + PDF report
    requirements.txt

------------------------------------------------------------------------

## Notes

-   CNN model is intentionally lightweight.
-   RandomForest captures frequency-domain camera fingerprints.
-   Registration uses ORB + RANSAC.
-   Designed for clarity, reproducibility, and modular structure.
