# Autism Detection Dashboard

A web-based dashboard for detecting autism from images using a trained deep learning model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your model file `BEST_MODEL_acc_0.9018_round_33.h5` is in the same directory.

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

1. Click "Browse files" to upload an image
2. Click "Analyze Image" to get the prediction
3. View the result showing whether the image indicates Autism or Non-Autism with confidence score

## Features

- Clean, user-friendly interface
- Real-time image upload and prediction
- Confidence scores and probability distribution
- Visual result display with color coding

## Note

This tool is for research purposes only and should not replace professional medical diagnosis.
