import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and accuracy
log_model = joblib.load("log_reg.pkl")
accuracy = joblib.load("model_accuracy.pkl")

# Set page config
st.set_page_config(page_title="💳 Credit Card Fraud Detection",
                   page_icon="💳", layout="wide")

# Custom CSS for gold theme and card style
st.markdown("""
    <style>
    .stApp {
        background-color: #FFD700;
        color: #000000;
    }
    .stSidebar {
        background-color: #F5DEB3;
        padding: 15px;
    }
    .stButton>button {
        background-color: #DAA520;
        color: white;
        font-weight: bold;
    }
    .card {
        background-color: #FFF8DC;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 5px 5px 10px #aaaaaa;
    }
    .stHeader, .stTitle {
        color: #8B4513;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
menu = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🔮 Single Prediction", "📂 Batch Prediction", "📊 Model Info"])
st.sidebar.markdown("---")

# Author info
st.sidebar.markdown("""
**Author:**  
Sherriff Abdul-Hamid  
**Email:**  
[sherriffhamid001@gmail.com](mailto:sherriffhamid001@gmail.com)  
**LinkedIn:**  
[LinkedIn](https://www.linkedin.com/in/sherriffhamid)
""")

# ======================
# Home Page
# ======================
if menu == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection App")
    st.markdown("""
    Welcome to the **Fraud Detection App**! 🚀  

    ### Features:
    - 🔮 **Single Prediction**: Enter transaction details manually.  
    - 📂 **Batch Prediction**: Upload a CSV with multiple transactions.  
    - 📊 **Model Info**: View performance metrics and visuals.  
    """)


# ======================
# Single Prediction
# ======================
elif menu == "🔮 Single Prediction":
    st.header("🔮 Single Transaction Prediction")
    st.markdown("Enter transaction details below to predict fraud (1) or legitimate (0).")

    with st.container():
        st.markdown('<div class="card"><h4>Transaction Details</h4></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=1.0, value=100.0)
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
        with col2:
            location = st.selectbox("Location", ["Domestic", "International"])
            device = st.selectbox("Device Type", ["Mobile", "Desktop", "ATM", "POS Terminal"])
            notes = st.text_input("Optional Notes")

    # Encode categorical features
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    type_map = {"Online": 0, "POS": 1, "ATM": 2}
    loc_map = {"Domestic": 0, "International": 1}
    dev_map = {"Mobile": 0, "Desktop": 1, "ATM": 2, "POS Terminal": 3}

    tval = time_map[time_of_day]
    ttyp = type_map[transaction_type]
    locv = loc_map[location]
    devv = dev_map[device]

    # Map to 24 features expected by model
    features = np.zeros(24)
    features[0] = amount / 1000
    features[1] = tval
    features[2] = ttyp
    features[3] = locv
    features[4] = devv
    features = features.reshape(1, -1)

    if st.button("Predict"):
        pred = log_model.predict(features)[0]
        proba = log_model.predict_proba(features)[0][1]

        if pred == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2f})")

        st.info(f"🎯 Model Accuracy: {accuracy:.2%}")

# ======================
# Batch Prediction
# ======================
elif menu == "📂 Batch Prediction":
    st.header("📂 Batch Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV File with Transaction Data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", data.head())

        if st.button("Run Batch Prediction"):
            preds = log_model.predict(data)
            data["Prediction"] = preds
            st.write("✅ Predictions Complete")
            st.dataframe(data.head(20))

            fraud_count = np.sum(preds)
            legit_count = len(preds) - fraud_count

            st.info(f"🚨 Fraudulent Transactions: {fraud_count}")
            st.success(f"✅ Legitimate Transactions: {legit_count}")

            # Visualization
            fig, ax = plt.subplots()
            sns.countplot(x=data["Prediction"], palette="coolwarm", ax=ax)
            ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
            st.pyplot(fig)

            # Save results
            data.to_csv("batch_predictions.csv", index=False)
            st.success("💾 Predictions saved as batch_predictions.csv")

# ======================
# Model Info
# ======================
elif menu == "📊 Model Info":
    st.header("📊 Model Performance")
    st.write(f"🎯 Logistic Regression Model Accuracy: **{accuracy:.2%}**")
    st.markdown("""
    **Model Details:**  
    - Algorithm: Logistic Regression  
    - Imbalance handled with `class_weight='balanced'`  
    - Metrics: ROC-AUC, Confusion Matrix, Accuracy  
    """)

    st.image("confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
    st.image("roc_curve.png", caption="ROC Curve", use_column_width=True)
