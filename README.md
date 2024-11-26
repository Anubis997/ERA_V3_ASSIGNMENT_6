# MNIST CNN Assignment

## Overview
This repository contains an implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model architecture includes several convolutional layers followed by a Global Average Pooling (GAP) layer.

## Model Architecture
- **Input Layer**: 1 channel (grayscale image)
- **Convolutional Layers**:
  - Conv Layer: 1 to 4 channels
  - Conv Layer: 4 to 8 channels
  - Conv Layer: 8 to 12 channels
  - Conv Layer: 12 to 16 channels
  - Conv Layer: 16 to 32 channels
  - Conv Layer: 32 to 16 channels
- **Global Average Pooling Layer**
- **Output Layer**: 10 classes (digits 0-9)

## Data Augmentation
The model uses the following data augmentation techniques:
- Random cropping
- Normalization

## Testing
The model has been tested for:
- Total parameter count (should be less than 20,000)
- Use of Batch Normalization
- Use of Dropout
- Use of either Fully Connected Layer or GAP (not both)

### Test Logs
You can view the test logs and results in the GitHub Actions section of this repository.

[Link to GitHub Actions](https://github.com/Anubis997/ERA_V3_ASSIGNMENT_7/actions)


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Anubis997/ERA_V3_ASSIGNMENT_7.git
   ```
2. Navigate to the directory:
   ```bash
   cd ERA_V3_ASSIGNMENT_7
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python Train.py
   ```

## License
This project is licensed under the MIT License.
