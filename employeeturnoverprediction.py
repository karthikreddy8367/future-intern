# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('/content/employee_data.csv')

# Print the column names
print(data.columns)

# Define the feature matrix X and the target vector y
X = data.drop('output', axis=1)  # replace 'output' with your actual target column
y = data['output']  # replace 'output' with your actual target column

# Drop rows with NaN values from the data
data.dropna(inplace=True)

# Redefine X and y after dropping NaN values
X = data.drop('output', axis=1)
y = data['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ]
)

# Create a pipeline with a scaler, a classifier, and the preprocessor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Take user input for various features
number = int(input("Enter the employee's number: "))
annual_salary = float(input("Enter the employee's annual salary: "))

# Create a new dataframe with the user input
new_data = pd.DataFrame({
    'number': [number],
    'annual_salary': [annual_salary],
    'first_name': [''],  # add empty strings for missing columns
    'last_name': [''],
    'gender': [''],
    'birth_date': [''],
    'employment_status': ['']
})

# Predict the output
y_pred = pipeline.predict(new_data)

print("Predicted output:", y_pred[0])
