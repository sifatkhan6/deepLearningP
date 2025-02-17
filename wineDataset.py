import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_train_cat = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_cat = encoder.transform(y_test.reshape(-1, 1))

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

# creating model
def create_model_a():
    """Create Model A: Single Hidden Layer"""
    model = Sequential([
        Dense(4, activation='relu', input_shape=(13,)),
        Dense(3, activation='softmax')
    ])
    return model


def create_model_b():
    """Create Model B: Two Hidden Layers"""
    model = Sequential([
        Dense(8, activation='relu', input_shape=(13,)),
        Dense(4, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model


def create_model_c():
    """Create Model C: Wide Single Layer"""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(13,)),
        Dense(3, activation='softmax')
    ])
    return model


def train_and_evaluate(model, lr, X_train, y_train, X_test, y_test):
    """Train and evaluate a model with given learning rate"""
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return history, test_loss, test_acc