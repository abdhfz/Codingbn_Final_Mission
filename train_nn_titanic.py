# train_nn_titanic.py
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Ensure model folder exists
os.makedirs('model', exist_ok=True)

# Load Titanic dataset
titanic = sns.load_dataset('titanic')
features = ['pclass','sex','age','sibsp','parch','fare','embarked']
X = titanic[features]
y = titanic['survived']

# Preprocessing pipeline
num_features = ['age','sibsp','parch','fare']
cat_features = ['pclass','sex','embarked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

X_processed = preprocessor.fit_transform(X)

# Save preprocessor
joblib.dump(preprocessor, 'model/titanic_preprocessor.joblib')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Neural Network
model = models.Sequential([
    layers.Input(shape=(X_processed.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])

# Save model
model.save('model/titanic_nn.h5')

print("Model and preprocessor saved successfully!")
