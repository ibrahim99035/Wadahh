#importing libraries
import pandas as pd
import numpy as np
import pickle as pk
#---------------------------------------------------------------------------------------------------------------------------------------------
#reading the data set in panadas dataFrame 
df = pd.read_csv('diabetes.csv')
#---------------------------------------------------------------------------------------------------------------------------------------------
# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})
#---------------------------------------------------------------------------------------------------------------------------------------------
#we should replace the 0 values in Glucose, BloodPressure, SkinThickness, Insulin, BMI with {NaN}
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#---------------------------------------------------------------------------------------------------------------------------------------------
#the next step is to fill null values with mean or median depending on distripution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)

df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)

df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)

df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)

df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)
#---------------------------------------------------------------------------------------------------------------------------------------------
#training the model :
from sklearn.model_selection import train_test_split
X = df.drop(columns = 'Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pk.dump(classifier, open(filename, 'wb'))

