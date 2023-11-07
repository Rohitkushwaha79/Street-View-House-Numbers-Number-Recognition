# Street View House Numbers Number Recognition

## Overview

This repository hosts a project for number recognition in street view house numbers. The primary goal of this project is to build and train a deep learning model capable of identifying digits in street view images.

### Key Features

- Robust digit recognition in street view images.
- Efficient preprocessing and data extraction techniques.
- Well-documented code.
- Easy-to-use model for recognizing house numbers.

## Table of Contents

1. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)

2. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
   - [Testing the Model](#testing-the-model)

3. [Contributing](#contributing)
   - [Issues](#issues)
   - [Pull Requests](#pull-requests)

5. [Test Results](#test-results)

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- Pandas
- NumPy

### Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/your-username/Street-View-House-Numbers_Number-Recognition.git

2. Navigate to the project directory:
   ```sh 
   cd Street-View-House-Numbers_Number-Recognition
3. Install the required dependencies:
   ```sh 
   pip install -r requirements.txt

## Usage
### Training the Model
To train the recognition model on your dataset, follow these steps:
Prepare your dataset and ensure it's structured correctly.
Update the dataset paths in the model training script.
Run the training script:
```sh
python train.py
```
### Evaluating the Model
You can evaluate the trained model's performance using a validation dataset:
```sh
python evaluate_model.py
```
### Testing the Model
Test the model's recognition accuracy on street view images:
```sh
python test_model.py /path/to/your/test/images
```
## Contributing
## Issues
If you encounter any issues or have suggestions, please create a GitHub issue.

## Pull Requests
We welcome pull requests for bug fixes or feature additions. Please ensure that your code complies with our guidelines.


## Test Results
![model_performance_plot](https://github.com/Rohitkushwaha79/Street-View-House-Numbers-Number-Recognition/assets/118690283/26e40b82-7fd5-4176-9ed0-61a9060c370b)

Here are the latest test results for the project:
- Training Accuracy: 96.7%
- Validation Accuracy: 92.4%
- Top-K Accuracy: 98.7%
- Top-K Val Accuracy: 96.7%
- Training Loss: 0.11
- Validation Loss: 0.27

On evaluating the model on the test dataset, it gives the following results:
- Test Loss: 0.23
- Test Accuracy: 93.7
- Top-K Accuracy: 97.5%

