# Bank Customer Churn Prediction  

This project predicts whether a bank customer is likely to **churn** (leave the bank) using machine learning. The goal is to help financial institutions identify at-risk customers and take proactive measures to improve retention.  

## 📂 Project Files  
- `Bank_Customer_Churn_Prediction.ipynb` → Jupyter notebook with the full workflow (data analysis, preprocessing, model training, and evaluation).  
- `Churn_Modelling.csv` → Dataset containing customer records used for model training.  

## 📊 Dataset Description  
The dataset contains **10,000 bank customers** with the following features:  
- **CustomerId, Surname** → Identification (not used in prediction).  
- **CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary** → Numerical inputs.  
- **Geography, Gender** → Categorical inputs.  
- **Exited** → Target variable (1 = customer churned, 0 = customer retained).  

## ⚙️ Project Workflow  
1. **Exploratory Data Analysis (EDA)** → Understanding distributions, correlations, and churn patterns.  
2. **Data Preprocessing** → Encoding categorical variables, feature scaling, and train-test split.  
3. **Model Training** → Logistic Regression, Random Forest, XGBoost, Neural Networks, and other algorithms.  
4. **Model Evaluation** → Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
5. **Feature Importance** → Identifying key drivers of churn.  

## 📈 Results  
- The best-performing model achieved strong accuracy and recall, balancing prediction power with interpretability.  
- Important features influencing churn include **Age, Geography, Balance, and Customer Activity Status**.  

## 🛠️ Tech Stack  
- **Python**  
  - Pandas, NumPy (data handling)  
  - Matplotlib, Seaborn (visualization)  
  - Scikit-learn (ML models & evaluation)  
  - TensorFlow/Keras, XGBoost (advanced models)  
- **Jupyter Notebook** for development and documentation  

## 🚀 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/sahilsable-24/Bank-Customer-Churn-Prediction.git
   cd Bank-Customer-Churn-Prediction
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook Bank_Customer_Churn_Prediction.ipynb
   ```  

## 📌 Future Enhancements  
- Hyperparameter tuning with GridSearchCV/Optuna  
- Deploy model as a **REST API** (Flask/FastAPI)  
- Interactive **Streamlit dashboard** for churn prediction  

## 📜 License  
This project is licensed under the **MIT License**.  
