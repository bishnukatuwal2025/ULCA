# UCLA Admission Prediction using Neural Networks

## Project Overview
This project builds a neural network classification model to predict whether a student has a high chance of admission to UCLA. The original notebook code is modularized into reusable Python files and structured into a VS Code project. The target variable `Admit_Chance` is converted into binary form using a threshold of 0.8.

## Dataset
The dataset contains 500 records with features such as GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research. There are no missing values. The dataset file `Admission.csv` should be placed inside the `data` folder.

## Project Structure
- `src/` contains modular code (data loading, preprocessing, model, evaluation, prediction, logging)
- `app/` contains the Streamlit application
- `data/` contains dataset
- `tests/` contains unit tests
- `main.py` is used for training
- `requirements.txt` contains dependencies

## Data Preprocessing
- Convert target variable to binary
- Drop unnecessary columns
- Apply one-hot encoding
- Perform train-test split with stratification
- Apply MinMaxScaler after split to avoid data leakage

## Model
The model is built using MLPClassifier with:
- hidden layer: (3,)
- activation: tanh
- max_iter: 200
- batch_size: 50

## Performance
- Train Accuracy: ~0.90  
- Test Accuracy: ~0.85–0.87  
The model performs well but does not fully reach 90 percent test accuracy.

## Logging and Error Handling
Logging is implemented using Python logging module with both console and file outputs. Error handling is included for data loading and prediction.

---

## Setup and Run Instructions

### 1. Create Virtual Environment

python -m venv .venv

### 2. Activate Enrironment

.venv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

### Train the model

python main.py

### Test the model 
pytest

### Streamlit Deploy

streamlit run app/streamlit_app.py