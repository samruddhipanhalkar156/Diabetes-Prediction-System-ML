# 🩺 Diabetes Prediction System  

## 🔹 Project Overview  
This project is a *machine learning-based diabetes prediction system* that predicts whether a patient has **Diabetes (1)** or is **Healthy (0)** based on health indicators from the **CDC Diabetes Health Indicators dataset (UCI ID: 891)**.  

### The workflow includes:  
1. **Data Cleaning & Preprocessing**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Scaling & Model Training** (Logistic Regression, SVM, Random Forest, XGBoost)  
4. **Model Evaluation & Comparison**  
5. **Interactive Web App using Streamlit**  

The app allows users to:  
✅ Select a trained model and view its performance metrics  
✅ Enter patient details manually for prediction  
✅ Upload a CSV file and download predictions  

```

## 🔹 Folder Structure  

Diabetes Prediction Analysis
│
├── Models
│ ├── LogisticRegression_diabetes.pkl
│ ├── RandomForest_diabetes.pkl
│ ├── SVM_diabetes.pkl
│ ├── XGBoost_diabetes.pkl
│ └── scaler.pkl
│
├── Model Results
│ ├── model_results_<timestamp>.xlsx
│
├── EDA Results
│ ├── correlation_heatmap.png
│ └── boxplots.png
│
├── Requirements
│ └── requirements.txt
│
├── main.py ← Script for training models
├── app.py ← Streamlit web app
└── README.md

```

## 🔹 Data Cleaning & Preprocessing  
- Removed **ID** column (not useful for prediction).  
- Target variable:  
  - **0 = No Diabetes**  
  - **1 = Prediabetes or Diabetes**  
- Features scaled using **StandardScaler**.  
- EDA performed with **correlation matrix, boxplots, and class balance visualization**.  



## 🔹 Column Descriptions  

| Column Name            | General Description |
|------------------------|----------------------|
| **Diabetes_binary**    | Target variable: 0 = No Diabetes, 1 = Diabetes/Prediabetes |
| **HighBP**             | High blood pressure (1 = Yes, 0 = No) |
| **HighChol**           | High cholesterol (1 = Yes, 0 = No) |
| **CholCheck**          | Cholesterol check in last 5 years (1 = Yes, 0 = No) |
| **BMI**                | Body Mass Index (numeric value) |
| **Smoker**             | Smoked at least 100 cigarettes in lifetime (1 = Yes, 0 = No) |
| **Stroke**             | History of stroke (1 = Yes, 0 = No) |
| **HeartDiseaseorAttack** | Coronary heart disease or heart attack (1 = Yes, 0 = No) |
| **PhysActivity**       | Physical activity in last 30 days (1 = Yes, 0 = No) |



## 🔹 Models Implemented  
1. **Logistic Regression** – Tuned with multiple solvers & regularization.  
2. **Support Vector Machine (SVM)** – Tested kernels (linear, RBF, poly).  
3. **Random Forest** – Optimized depth & number of estimators.  
4. **XGBoost** – Tuned learning rate, max depth, and subsampling.  

All models are saved as `.pkl` files for reuse.  


## 🔹 Evaluation Metrics  
Each model is evaluated on *train and test sets* using:  
- `Accuracy`  
- `Precision`  
- `Recall`  
- `F1-score`  
- `AUC-ROC`  

📊 Results are saved in: **Model Results/model_results_<timestamp>.xlsx**  



## 🔹 Streamlit Web App (app.py)  
Features:  
- Dropdown for **model selection** (`Logistic Regression`, `SVM`, `RF`, `XGBoost`).  
- Displays **model performance metrics** from Excel.  
- Manual input → Predicts **Healthy / Diabetes**.  
- CSV upload → Generates **batch predictions + download option**.  



## 🔹 How to Run  

### 1️⃣ Install Dependencies  
```bash
cd "Diabetes Prediction Analysis"
pip install -r Requirements/requirements.txt
```
2️⃣ Run Training Script
```bash

python main.py
```
3️⃣ Run Streamlit App
```
streamlit run app.py
Then open: http://localhost:8501 in your browser.
```

## 🔹 Requirements
See requirements.txt for the full list.

Key packages:

`pandas`, `numpy`

`scikit-learn`

`matplotlib`, `seaborn`

`xgboost`

`streamlit`

`openpyxl`

## 🔹 Future Enhancements
Add deep learning models (ANN, CNN)

Feature importance visualization (SHAP, LIME)

Deploy on Streamlit Cloud / Heroku

API integration for hospital data systems

## 🔹 Author

👩‍💻 **amruddhi Panhalkar**

📧 Email: samruddhipanhalkar156@gmail.com

🏫 Designation: Robotics and Artificial Intelligence

🌐 LinkedIn
