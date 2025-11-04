
# ❤️ Heart Disease Prediction Project

This project is about predicting whether a person is likely to have heart disease using machine learning.
I used a dataset that contains information such as age, blood pressure, cholesterol level, and other medical details of patients.

The main idea is to train a few machine learning models and find which one gives the best accuracy in predicting heart disease.

---

## 🧠 What I Did

1. **Loaded the dataset** using pandas and explored the data to understand it better.
2. **Cleaned and prepared the data** — checked for missing values and made sure everything was ready for modeling.
3. **Analyzed the data** using graphs and charts with Matplotlib and Seaborn to see patterns and relationships between features.
4. **Split the data** into training and testing sets.
5. **Trained different machine learning models** like Logistic Regression, Random Forest, and SVM.
6. **Evaluated the models** using accuracy, precision, recall, and F1 score.
7. **Compared results** and visualized them using confusion matrices and heatmaps.

---

## 🧩 Tools and Libraries Used

* **Python**
* **Pandas** and **NumPy** for data handling
* **Matplotlib** and **Seaborn** for data visualization
* **Scikit-learn** for building machine learning models
* **Jupyter Notebook** for writing and running the code

---

## 📊 About the Dataset

The dataset includes medical details such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* ECG results
* Maximum heart rate achieved
* Exercise-induced angina
* And the target column (1 = has heart disease, 0 = no heart disease)

---

## 🔍 Results and Observations

* Random Forest performed the best among the models I tested.
* Features like **age**, **chest pain type**, and **maximum heart rate** were highly related to heart disease chances.
* The project showed how data and machine learning can help predict serious health risks early.

---

## 📁 Folder Structure

```
HEART_DISEASE_PROJECT/
│
├── data/
│   └── heart.csv                # dataset
│
├── notebooks/
│   ├── heart_analysis.ipynb     # main Jupyter notebook (code + visuals)
│   └── heart_disease_model.pkl  # saved trained model file
│
└── README.md                    # project summary/documentation


## ⚙️ How to Run

1. Open the project folder in **VS Code** or **Jupyter Notebook**.
2. Make sure the required libraries are installed:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run each cell in the notebook one by one to see the analysis and results.



## 🙋 About Me

**Name:** Rabiya Azami
**Role:** Data Science Intern
**Goal:** To build practical projects in data science and machine learning.


Comparative Analysis

To evaluate the effectiveness of the proposed Logistic Regression model, we compared its performance with two other widely used machine learning algorithms — Decision Tree and K-Nearest Neighbors (KNN) — on the same dataset.

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	85.25%	84%	87%	85%
Decision Tree	79.34%	78%	80%	79%
K-Nearest Neighbors	81.96%	80%	82%	81%

Conclusion:
Logistic Regression performed best among the tested models, demonstrating higher accuracy and balanced performance across all metrics. It also offered faster computation time and better generalization on unseen data. Hence, it was chosen as the final model for the proposed heart disease prediction system.