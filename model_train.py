import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load your dataset
df = pd.read_csv('data/data_final.csv')

# Splitting data into features and target
X = df[['Feature_1', 'Feature_2']]  # Features
y = df['Target']                    # Target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create a directory for the model if it doesn't exist
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Saving the model to the 'model' folder
model_path = os.path.join(model_dir, 'model.pkl')
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
