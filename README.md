
# Neural Network Model Builder

This project provides a step-by-step guided system for building and training a neural network using Python and Jupyter notebooks. It allows users to upload their own dataset or use built-in datasets, preprocess the data, split it into training and testing sets, and configure a neural network model for training.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Project Steps](#project-steps)
    - [Step 1: Data Collection](#step-1-data-collection)
    - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
    - [Step 3: Train-Test Split](#step-3-train-test-split)
    - [Step 4: Model Training](#step-4-model-training)
5. [Acknowledgments](#acknowledgments)

## Installation

To run this project, you need to have the following libraries installed:

- `pandas`
- `scikit-learn`
- `ipywidgets`
- `matplotlib`
- `numpy`
- `keras`

You can install these libraries using pip:

```bash
pip install pandas scikit-learn ipywidgets matplotlib numpy keras
```

## Usage

1. Open the Jupyter notebook containing the project code.
2. Follow the steps outlined in the notebook to load your dataset, preprocess the data, split the data, and train the model.
3. The trained model and evaluation metrics will be displayed after training.

## Features

- Upload your own CSV dataset or use built-in datasets (Iris, Wine, California Housing, Breast Cancer, Diabetes).
- Preprocess data by handling missing values, encoding categorical variables, and applying scaling.
- Configure and train a neural network model with customizable layers, units, activation functions, optimizer, and regularization.
- View training and validation metrics, including loss and accuracy plots.

## Project Steps

### Step 1: Data Collection

In this step, you can either upload your own dataset or select a built-in dataset from the dropdown menu.

- **Upload CSV**: Upload your CSV file using the upload button.
- **Select Built-in Dataset**: Choose a dataset from the dropdown menu (Iris, Wine, California Housing, Breast Cancer, Diabetes).

After loading the dataset, basic information about the dataset, including the number of instances, features, and the first few rows, will be displayed.

### Step 2: Data Preprocessing

Preprocess the data by selecting features, target columns, and various preprocessing methods.

- **Select Feature**: Choose the feature to visualize and use for modeling.
- **Select Target**: Choose the target column.
- **Preprocess**: Choose a preprocessing method (Standard Scaler, Min-Max Scaler, Robust Scaler).
- **Drop Columns**: Select columns to drop from the dataset.
- **Fill Missing Values**: Select columns to fill missing values and choose the fill method (Mean, Median, Mode).
- **Handle Categorical Data**: Select categorical columns and choose the encoding method (Label Encoding, One-Hot Encoding).

### Step 3: Train-Test Split

Split the data into training and testing sets using an 80-20 split.

- **Train-Test Split**: Click the button to split the data.

### Step 4: Model Training

Configure and train a neural network model.

- **Number of Layers**: Select the number of layers in the neural network.
- **Units in Each Layer**: Select the number of units in each layer.
- **Activation Functions**: Choose the activation function for each layer.
- **Optimizer**: Select the optimizer (SGD, Adam, RMSprop).
- **Regularization**: Select regularization (None, L1, L2).
- **Epochs**: Select the number of epochs for training.

- **Train Model**: Click the button to train the model. The training and validation metrics, including loss and accuracy, will be displayed along with classification report and plots.

## Acknowledgments

This project uses the following libraries and datasets:

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [Keras](https://keras.io/)
- Built-in datasets from Scikit-learn and OpenML

