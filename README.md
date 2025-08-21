# california-house-price-prediction
# California House Price Prediction

A machine learning project to **predict house prices in California** based on various input features.  
This project uses a trained model and preprocessing pipeline to provide accurate predictions.

---

## Project Structure

| File/Folder       | Description |
|------------------|-------------|
| `main.py`         | Main script to run house price predictions |
| `main_old.py`     | Older version of the main script for reference |
| `predict.py`      | Script for predicting using command-line inputs |
| `predict_tk.py`   | Script for GUI-based predictions using Tkinter |
| `app.py`          | Optional Streamlit or Flask app interface for web-based predictions |
| `housing.csv`     | Dataset used for training the model (optional for reference) |
| `input.csv`       | Sample input data for predictions |
| `output.csv`      | Sample output predictions |
| `model.pkl`       | Trained machine learning model (**not included in repo, download separately**) |
| `pipeline.pkl`    | Preprocessing pipeline (**not included in repo, download separately**) |
| `.gitignore`      | Ignores large model files to keep repo under 100 MB |
| `README.md`       | Project documentation |

---

## Features

- Predict California house prices using input features
- Uses trained model (`model.pkl`) and preprocessing pipeline (`pipeline.pkl`)
- Supports both command-line and GUI-based predictions
- Can be extended to other datasets

---

## 🔽 Download the Models

To run predictions, you'll need to download the trained model and preprocessing pipeline. These files are not included in the repository due to size constraints.

### 📁 Files to Download:
- [`model.pkl`](https://drive.google.com/file/d/1cUe2WADBh9-QeGmKsgl9xJpBtKWS93vS/view?usp=sharing) – Trained machine learning model  
- [`pipeline.pkl`](https://drive.google.com/file/d/1TvWSbniMF3vhlR78qIKlWvk5fzw3BRWa/view?usp=sharing) – Preprocessing pipeline used during training

### 📦 Instructions:
1. Download both files from the links above.
2. Place them in the root directory of this project (same level as `main.py`).
3. Ensure filenames remain unchanged (`model.pkl` and `pipeline.pkl`).

> ⚠️ If you rename or relocate the files, update the file paths in your scripts accordingly.

---

## Requirements

- Python 3.8+  
- Required packages (install via pip):

```bash
pip install -r requirements.txt
