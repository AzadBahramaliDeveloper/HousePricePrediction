# House Price Prediction Model Report

## 1. Data Preparation

The dataset was initially loaded from two CSV files: `train.csv` and `test.csv`, using pandas' `read_csv()` method. A brief exploration of the dataset was performed using `info()` and `describe()` to gain insights into the data structure and statistical summary.

Missing values were checked using `isnull().sum()`. The dataset was cleaned by:
- **Numerical Columns**: Missing values in numerical columns were filled with the mean of the respective column using `fillna()`.
- **Categorical Columns**: Missing values in categorical columns were filled with the most frequent value (mode) using `fillna()`.

Next, the categorical columns `POSTED_BY` and `BHK_OR_RK` were handled through **one-hot encoding** using pandas’ `get_dummies()` method, which converts categorical variables into binary columns. The `ADDRESS` column was dropped as it was deemed irrelevant for predicting the target variable.

The dataset was then scaled to standardize numerical features such as `SQUARE_FT`, `LONGITUDE`, and `LATITUDE` using **StandardScaler** from scikit-learn. This scaling step ensures that all features are on the same scale, which is important for some machine learning models.

---

## 2. Model Selection

For the model, the **Random Forest Regressor** was chosen. This decision was based on several factors:
- **Random Forests** are **ensemble models** that combine multiple decision trees to improve prediction accuracy and avoid overfitting.
- They are **non-linear models**, which allows them to capture complex relationships in the data, making them suitable for predicting house prices, which are influenced by multiple factors.
- Random Forests also handle both **numerical and categorical data** well without needing explicit feature scaling (although scaling was done for consistency).

---

## 3. Model Performance

The model was trained on the training set (`X_train` and `y_train`), and its performance was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: The average of the absolute differences between the predicted and actual values.
- **Mean Squared Error (MSE)**: The average of the squared differences between the predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which provides a measure of the average prediction error in the same units as the target variable.
- **R² (Coefficient of Determination)**: This metric indicates how well the model explains the variance in the target variable. An **R² value of 0.74** was achieved, meaning that the model explains **74%** of the variance in house prices.

The performance of the model was deemed **satisfactory**, as the R² score was above **0.7**, indicating that the model has reasonable predictive power. However, there is still room for improvement, especially in the areas of feature engineering and model fine-tuning.

---

## 4. Suggestions for Improvement

While the Random Forest model performed well, several improvements could be made:
- **Feature Engineering**: Adding more features, such as the interaction between existing features, could potentially improve the model’s accuracy. For example, combining `SQUARE_FT` with `BHK_NO.` might provide more insights into the house size and pricing.
- **Model Alternatives**: Exploring other models, such as **Gradient Boosting** or **XGBoost**, which have shown strong performance in many regression tasks, could lead to improved results.
- **Hyperparameter Tuning**: Although Random Forest was used with default hyperparameters, using techniques like **GridSearchCV** or **RandomizedSearchCV** (as performed in this case) could help optimize the model’s parameters further. The best parameters were found using `RandomizedSearchCV`, and a model was refitted using these parameters.

---

## 5. Conclusion

In conclusion, the **Random Forest Regressor** provided satisfactory results in predicting house prices, with an **R² value of 0.74**. The model demonstrated good generalization capabilities, though improvements can be made through better feature engineering and further exploration of alternative models. The analysis provided insights into how feature scaling, missing data handling, and categorical encoding contribute to the performance of machine learning models.

---

## Code Files and Submission

The full code for the model can be found in the Python script named **`house_price_prediction.py`**, which contains the data preparation, model training, evaluation, and tuning processes.

Additionally, the **best model** after hyperparameter tuning was saved as **`house_price_model_best.pkl`** for future predictions.

---

### **Acknowledgments**

Thanks to the libraries and tools used in this project:
- **pandas** for data manipulation
- **scikit-learn** for machine learning algorithms and model evaluation
- **matplotlib** for visualizing the model performance

---
# For VG

## 1. Data Preparation

This section remains unchanged from the **Godkänd** documentation. It describes how the data was loaded, cleaned, and prepared for modeling. Missing values were handled, categorical variables were one-hot encoded, and numerical features were standardized using **StandardScaler**.

---

## 2. Model Selection

Initially, the **Random Forest Regressor** was chosen as the model due to its robustness and ability to handle both numerical and categorical features. It is an ensemble model that combines multiple decision trees to improve prediction accuracy.

For the **Väl Godkänd (VG)** level, an additional approach was explored using **Logistic Regression**. Logistic Regression was chosen for its computational efficiency and interpretability. This model was further enhanced by creating **polynomial features** of degree 2, which were used to capture non-linear relationships between the variables. The creation of polynomial features allowed the model to better handle complex interactions in the data.

Additionally, **hyperparameter tuning** was performed on the Logistic Regression model using **GridSearchCV** to identify the best set of hyperparameters. The hyperparameters tuned included:
- `C` (regularization strength)
- `solver` (optimization algorithm)
- `max_iter` (number of iterations)

These hyperparameters were tuned using **cross-validation** to ensure the model’s generalization capability. The best parameters were selected, and a new Logistic Regression model was trained with the optimal configuration.

---

## 3. Model Performance

For the **Väl Godkänd** level, **Logistic Regression** with **polynomial features** was tested alongside the Random Forest model. The performance of the Logistic Regression model was evaluated using **accuracy** and **classification report**.

The following steps were performed to evaluate the models:
- The models were evaluated using **accuracy** to understand the overall correctness of predictions.
- A **classification report** was generated to assess precision, recall, F1-score, and support for each class (price range categories).

It was found that the Logistic Regression model with polynomial features provided competitive accuracy while being computationally more efficient and interpretable compared to Random Forest.

---

## 4. Hyperparameter Tuning

For the **Väl Godkänd (VG)** level, **GridSearchCV** was used to tune the hyperparameters of the Logistic Regression model. The following parameters were optimized:
- `C` (regularization strength)
- `solver` (optimization algorithm)
- `max_iter` (number of iterations)

The tuning process was carried out using **5-fold cross-validation** to select the best hyperparameters and prevent overfitting. The best parameters were then used to train the final model, which was then evaluated on the test data.

---

## 5. Polynomial Feature Transformation

To improve model performance, **polynomial features** of degree 2 were created. These features captured non-linear relationships between the variables, which could improve the model's predictive power. The polynomial features
