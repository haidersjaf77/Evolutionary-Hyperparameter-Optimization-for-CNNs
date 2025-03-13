import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load and preprocess the CIFAR-10 dataset
(X, y), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to range [0, 1]
X = X.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)
y_test = encoder.transform(y_test)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evolutionary algorithm components
def initialize_population(size, input_dim, output_dim):
    population = []
    for _ in range(size):
        individual = {
            'layers': np.random.randint(1, 6),
            'neurons': [np.random.randint(32, 128) for _ in range(np.random.randint(1, 6))],
            'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
            'learning_rate': 10 ** np.random.uniform(-4, -2),
            'batch_size': np.random.choice([32, 64, 128]),
            'dropout': np.random.uniform(0, 0.5)
        }
        population.append(individual)
    return population

def evaluate_individual(individual):
    model = tf.keras.Sequential()

    # Ensure the input shape is correct for CIFAR-10 (32x32x3)
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))

    for idx, neurons in enumerate(individual['neurons']):
        model.add(tf.keras.layers.Conv2D(neurons, (3, 3), activation=individual['activation'], padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Apply dropout if specified
        if individual['dropout'] > 0:
            model.add(tf.keras.layers.Dropout(individual['dropout']))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=individual['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to reduce computational cost
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    start_time = time.time()
    model.fit(X_train, y_train, epochs=3, batch_size=individual['batch_size'],
              validation_data=(X_val, y_val), verbose=2, callbacks=[early_stop])
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

    return accuracy, precision, recall, f1, training_time, y_test_classes, y_pred_classes