# Evolutionary Hyperparameter Optimization for CNNs  

## Overview  
This project applies **Evolutionary Algorithms** to optimize **Convolutional Neural Networks (CNNs)** for **image classification** using the **CIFAR-10 dataset**. By evolving hyperparameters such as the number of layers, activation functions, learning rates, and dropout rates, the algorithm aims to improve model performance compared to manually tuned CNNs.  

## Features  
- **Dataset Preprocessing**: Normalizes images and applies one-hot encoding.  
- **Manual CNN Model**: Predefined architecture with fixed hyperparameters.  
- **Evolutionary Algorithm**:  
  - Randomly initializes a population of CNN models.  
  - Selects top-performing models for crossover and mutation.  
  - Iteratively improves architecture through generations.  
- **Performance Comparison**: Compares manually tuned CNNs with evolved models using **accuracy, precision, recall, and F1-score**.  
- **Visualization**: Displays results using **bar charts and performance metrics**.  

## Technologies Used  
- **Python** (NumPy, pandas, TensorFlow/Keras, scikit-learn)  
- **Machine Learning & Neural Networks**  
- **Evolutionary Computing**  

## How to Run  
1. Install dependencies:  
   ```sh
   pip install numpy pandas tensorflow scikit-learn matplotlib
