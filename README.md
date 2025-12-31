# Medical-Insuarance-Cost-Predecition
<img width="1248" height="883" alt="image" src="https://github.com/user-attachments/assets/a51d7ba4-ef37-4337-a51c-442bbf392167" />


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview
This project aims to predict individual medical costs billed by health insurance using Machine Learning. By analyzing demographic and health-related factors, the model helps determine how different variables such as age, BMI, and lifestyle choices impact insurance premiums.

## 📊 Dataset Features
The dataset contains several key features that influence medical charges:

| Feature | Description |
| :--- | :--- |
| **Age** | Age of the primary beneficiary. |
| **Sex** | Gender of the insurance contractor (male/female). |
| **BMI** | Body Mass Index ($kg/m^2$), providing an index of weight-to-height ratio. |
| **Children** | Number of children/dependents covered by the insurance. |
| **Smoker** | Whether the person smokes or not (Significant factor). |
| **Region** | Residential area in the US (Northeast, Southeast, Southwest, Northwest). |
| **Charges** | **Target Variable:** Individual medical costs billed by health insurance. |



## ⚙️ Workflow
1. **Data Preprocessing:** Handling categorical variables through Label Encoding or One-Hot Encoding.
2. **Exploratory Data Analysis (EDA):** Checking correlations between features (e.g., how smoking impacts charges).
3. **Train-Test Split:** Splitting the data to evaluate model performance on unseen data.
4. **Model Building:** Implementing Regression algorithms (Linear Regression, Random Forest, etc.).
5. **Evaluation:** Assessing performance using metrics like $R^2$ Score and Mean Absolute Error (MAE).



## 🛠️ Installation
To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Medical-Insurance-Cost-Prediction.git](https://github.com/your-username/Medical-Insurance-Cost-Prediction.git)
