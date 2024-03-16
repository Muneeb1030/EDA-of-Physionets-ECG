import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('dataset.csv', header=None, names=['id', 'age', 'gender', 'dx'])

# Convert 'age' column to numeric type, handling non-numeric values
data['age'] = pd.to_numeric(data['age'], errors='coerce')

# Impute missing values in the 'age' column with median
imputer = SimpleImputer(strategy='median')
data['age'] = imputer.fit_transform(data[['age']])

# Verify if all missing values are filled
print("Number of missing values in 'age' column after filling:", data['age'].isnull().sum())
print((data['gender']=="Unknown").sum())
print((data['age']==0).sum())

with open('status1.txt',"w") as f:
    f.write(str(data.isnull().sum()))

data.to_csv("dataset1.csv",index=False)
