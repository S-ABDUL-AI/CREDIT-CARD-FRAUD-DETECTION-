# CREDIT-CARD-FRAUD-DETECTION-
An app that predicts whether a credit card transaction is fraudulent or authentic
A Streamlit-based web application** to detect fraudulent credit card transactions using Logistic Regression. This app allows single transaction prediction, batch prediction using CSV files, and provides model performance visualization.
## 🏠 Features
- 🔮 **Single Prediction**: Enter transaction details manually to check if a transaction is fraudulent.  
- 📂 **Batch Prediction**: Upload a CSV file containing multiple transactions for batch fraud detection.  
- 📊 **Model Info**: View performance metrics, including **Confusion Matrix**, **ROC Curve**, and **Model Accuracy**.  
- 💡 Handles imbalanced datasets with class weighting (`class_weight='balanced'`).  

FILE STRUCTURE 
├── app.py                # Streamlit app
├── log_reg.pkl           # Trained Logistic Regression model
├── model_accuracy.pkl    # Model accuracy
├── confusion_matrix.png  # Confusion matrix image
├── roc_curve.png         # ROC curve image
├── requirements.txt      # Python dependencies
└── README.md             # This file

Author
Author:
Sherriff Abdul-Hamid
Email:
sherriffhamid001@gmail.com
