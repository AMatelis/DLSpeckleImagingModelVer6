# Speckle-Based Flow Rate Estimation (3D Linear Regression Model)

This project demonstrates how deep learning can predict microfluidic flow rates in microliters per minute (µL/min) from grayscale speckle pattern videos. It uses a 3D Convolutional Neural Network (BloodFlowCNN) to analyze sequences of speckle frames and estimate the continuous flow rate. This work was developed under the guidance of Dr. Christopher Raub as part of a study on non-invasive optical methods for measuring blood flow.

# Background

Speckle patterns occur when coherent light, such as a laser, scatters through dynamic and heterogeneous media like flowing blood. Changes in flow speed or direction cause the speckle patterns to evolve over time. By analyzing sequences of these patterns, the underlying flow rate can be predicted accurately using a deep learning model.

# Project Overview

This project:

Loads AVI speckle videos named according to flow rate (e.g., 5ulpermin.avi).

Splits each video into overlapping grayscale frame stacks.

Trains the BloodFlowCNN 3D model to predict flow rate from each stack.

Outputs predictions to CSV files and visualizes results.

Supports evaluation using saved checkpointed weights.

# Folder Structure
project-root/
│
├── data/                  # Input AVI videos (*.avi)
├── models/
│   └── bloodflow_cnn.py   # Core 3D CNN model
├── src/
│   ├── dataset.py         # Custom PyTorch Dataset
│   ├── dataloader.py      # Preprocessing and data loader functions
│   └── train.py           # Training pipeline
├── outputs/
│   ├── predictions.csv    # CSV with predicted flow rates
│   ├── flowrate_plot.png  # True vs predicted flow rates
│   ├── train_loss.png     # Training and validation loss curves
│   ├── scaler.pkl         # Saved target normalizer
│   └── checkpoints/       # Folder for saved model weights
├── main.py                # CLI entry point for training or evaluation
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

# How to Use
Install Dependencies
pip install -r requirements.txt

Prepare Your Data

Place all input AVI files in the data/ folder. Expected filenames:

0ulpermin.avi

5ulpermin.avi

50ulpermin.avi

100ulpermin.avi

150ulpermin.avi

400ulpermin.avi

Each video should contain at least five frames (default sequence length).

Train the Model

Run:

python main.py --mode train


This will:

Train BloodFlowCNN for the specified number of epochs.

Save the best model to outputs/checkpoints/.

Save predictions to outputs/predictions.csv.

Plot predicted vs true flow rates to outputs/flowrate_plot.png.

Save training and validation loss curves to outputs/train_loss.png.

Evaluate a Trained Model

Run:

python main.py --mode evaluate --checkpoint outputs/checkpoints/best_model.pth


This reloads the model and reports regression metrics such as:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

R² Score (Explained Variance)

Predictions are saved to outputs/predictions.csv, including:

# Index

True flow rate

Predicted flow rate

Absolute error

Squared error

Relative error (%)

Example row from predictions:

0, 400.0, 380.42, 19.58, 383.29, 4.9

# Requirements

Python packages required:

torch

torchvision

opencv-python

scikit-learn

matplotlib

pandas

notebook

# Acknowledgement

Developed under the guidance of Dr. Christopher Raub with assistance from Thuc Pham at The Catholic University of America. This project investigates speckle-based imaging techniques for non-invasive blood flow estimation in microfluidic systems.
