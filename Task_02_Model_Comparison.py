import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("D:\Apna clg\internship\Task_02_Model_Comparison\dataset\Placement_Data_Full_Class.csv")

print(df.head())

# Check dataset shape
print("Dataset Shape:", df.shape)

# Check columns
print("\nDataset Columns:")
print(df.columns.tolist())

# Check data types and null values
print("\nDataset Information:")
df.info()

print(df['status'].value_counts())

# Drop unnecessary columns
df = df.drop(columns=['sl_no', 'salary'])

# Independent variables
X = df.drop('status', axis=1)

# Dependent variable
y = df['status']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

print("Missing Values in X:")
print(X.isnull().sum())

print("\nMissing Values in y:")
print(y.isnull().sum())

numerical_columns = [
    'ssc_p',
    'hsc_p',
    'degree_p',
    'etest_p',
    'mba_p'
]
categorical_columns = [
    'gender',
    'ssc_b',
    'hsc_b',
    'hsc_s',
    'degree_t',
    'workex',
    'specialisation'
]

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

print("\nTotal Numerical Columns:", len(numerical_columns))
print("Total Categorical Columns:", len(categorical_columns))

print("Numerical Data Types:")
print(X[numerical_columns].dtypes)

print("\nCategorical Data Types:")
print(X[categorical_columns].dtypes)

# Create preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),

        ('cat', OneHotEncoder(handle_unknown='ignore'),
         categorical_columns)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

models = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Decision tree": DecisionTreeClassifier(random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    pipeline.fit(X_train, y_train)
    trained_models[name]=pipeline
    
    print(f"{name} trained successfully!!!")
    
results = []

for name, pipeline in trained_models.items():
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Placed')
    recall = recall_score(y_test, y_pred, pos_label='Placed')
    f1 = f1_score(y_test, y_pred, pos_label='Placed')
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    
comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by='F1 Score',
    ascending=False
)

comparison_df
    
comparison_df.set_index('Model')[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score']
].plot(
    kind='bar',
    figsize=(12, 6)
)

plt.title('Machine Learning Model Comparison')
plt.ylabel('Score')
plt.xlabel('Algorithms')
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['Not Placed', 'Placed']
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Placed', 'Placed'],
        yticklabels=['Not Placed', 'Placed']
    )

    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()
    
best_model = comparison_df.iloc[0]

print("Best Performing Model:", best_model['Model'])
print("Accuracy:", best_model['Accuracy'])
print("Precision:", best_model['Precision'])
print("Recall:", best_model['Recall'])
print("F1 Score:", best_model['F1 Score'])