import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

## read the dataset
dataset= pd.read_csv('data.csv')

## deleting the column
dataset = dataset.drop(columns='Unnamed: 32')

## there are two values only, we will encode them into 1 and 0 (one hot encoding)
dataset= pd.get_dummies(data=dataset, drop_first=True)

dataset2 = dataset.drop(columns='diagnosis_M')

# matrix of features/ independent variables
x= dataset.iloc[:,1:-1].values

# dependent / target variable
y = dataset.iloc[:,-1]

## x is the independent variables and y the dependent variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

### Feature scalling / Normalization
sc = StandardScaler()

x_train= sc.fit_transform(x_train)
x_test=sc.transform(x_test)

## building the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x, y)

feature_importances = model.feature_importances_
feature_importances
importance_df = pd.DataFrame({'feature':  dataset.iloc[0, 1:-1].tolist() , 'importance': feature_importances})
top_features = importance_df.nlargest(7, 'importance')

selected_column_indices = [22, 20, 27, 23, 7, 13, 3]
selected_columns = dataset.iloc[:, selected_column_indices]

target_column = 'diagnosis_M'

# training
training_columns = ['texture_worst', 'fractal_dimension_se', 'concavity_mean', 'perimeter_worst', 'concavity_worst', 'perimeter_se', 'perimeter_mean']

X = dataset[training_columns]
Y = dataset[target_column]

# Split your data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train your Logistic Regression model
model1 = LogisticRegression()
model1.fit(X_train, y_train)

def predict_model(input_data):
    input_data = [input_data]
    prediction = model1.predict(input_data)
    if (prediction == 1):
        return 'Malignant'
    elif prediction == 0:
        return 'Benign'



