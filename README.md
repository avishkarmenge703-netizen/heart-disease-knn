# Heart Disease Prediction using KNN

This project uses a K-Nearest Neighbors (KNN) classifier to predict the presence of heart disease based on medical attributes. A Streamlit web app provides an interactive interface for users to input data and get instant predictions.

## Dataset
- **Source**: Cleveland Heart Disease dataset (UCI)
- **File**: `heart.csv` included in this repository (303 samples, 13 features)
- **Target**: 1 = disease, 0 = no disease

## Features
- **Algorithm**: KNN with k=5
- **Accuracy**: ~85% on test set
- **Web App**: Built with Streamlit, deployed on Streamlit Community Cloud

## Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-knn.git
   cd heart-disease-knn
