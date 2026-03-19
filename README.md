# 📊 Customer Churn Prediction System

## 🚀 Overview
This project predicts whether a customer is likely to churn using machine learning techniques. It analyzes customer behavior and provides real-time predictions through an interactive web dashboard.

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Web App:** Streamlit  

---

## 📂 Project Structure


Customer-Churn-Prediction/
├── data/
│ └── churn.csv
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
├── model/
│ ├── churn_model.pkl
│ ├── columns.pkl
├── notebooks/
│ └── eda.ipynb
├── app.py
├── requirements.txt
├── README.md


---

## 📊 Features
✔ Data Cleaning & Preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Machine Learning Model (Random Forest)  
✔ ~85% Prediction Accuracy  
✔ Interactive Streamlit Dashboard  
✔ Real-time Prediction with Probability  

---

## 🧠 Key Insights
- Customers with **higher monthly charges** are more likely to churn  
- **Short tenure customers** have higher churn risk  
- **Month-to-month contracts** increase churn probability  

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Train the model
python src/train_model.py

3️⃣ Run the application
streamlit run app.py
