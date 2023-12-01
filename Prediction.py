# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the CSV file
df = pd.read_csv('Housing.csv')

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
df['mainroad'] = label_encoder.fit_transform(df['mainroad'])
df['guestroom'] = label_encoder.fit_transform(df['guestroom'])
df['basement'] = label_encoder.fit_transform(df['basement'])
df['hotwaterheating'] = label_encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = label_encoder.fit_transform(df['airconditioning'])
df['prefarea'] = label_encoder.fit_transform(df['prefarea'])
df['furnishingstatus'] = label_encoder.fit_transform(df['furnishingstatus'])

# Separate features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model for later use
import joblib
joblib.dump(model, 'house_price_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')