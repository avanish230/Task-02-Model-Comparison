# Task 02 – Train and Compare Machine Learning Models

## Project Overview
This project focuses on training and comparing three Machine Learning classification algorithms for Student Placement Prediction.

The objective is to evaluate different algorithms using the same dataset and train-test split.

## Dataset
- Dataset: Student Placement Dataset
- Total Records: 215
- Features: 12 (after removing unnecessary columns)
- Target Variable: `status`

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Data Preprocessing
- Removed unnecessary columns (`sl_no`, `salary`).
- Separated input features and target variable.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Used ColumnTransformer and Pipeline.

## Train-Test Split
- Training Data: 80%
- Testing Data: 20%
- Random State: 42
- Stratified Sampling: Yes

The same train-test split was used for all three models to ensure a fair comparison.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Comparison
The performance of all three algorithms was compared using the above evaluation metrics.

The best-performing model was identified based on its F1-Score, while also considering Precision, Recall, and Confusion Matrix.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- VS Code

## How to Run

1. Clone this repository.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python file or open the Jupyter Notebook, depending on the project format.

## Conclusion
This project helped me understand how different classification algorithms perform on the same dataset and why evaluating models using multiple metrics is important.

## Author
Avanish