# Mushroom Classification Project

This project focuses on classifying mushrooms as either edible or poisonous using a Convolutional Neural Network (CNN) based on the InceptionV3 architecture. The model is trained and fine-tuned using data augmentation techniques and later exported as a TensorFlow Lite model for deployment on mobile or embedded devices.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Training, Fine-Tuning, and Exporting](#model-training-fine-tuning-and-exporting)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify mushrooms as edible or poisonous. This is crucial for preventing mushroom poisoning, which can have serious health consequences. The InceptionV3 model is used as the base model due to its robust architecture, and additional layers are added to improve classification performance. The project involves data preprocessing, model training with data augmentation, fine-tuning, evaluation, and model deployment.

## Dataset

The dataset used in this project is obtained from Kaggle. It contains images of mushrooms categorized into edible and poisonous classes. The dataset is divided into training and validation sets to ensure the model can generalize well to new, unseen data. Data augmentation techniques such as rotation, width shift, height shift, shear, zoom, and horizontal flip are applied to the training images to make the model robust to various transformations.

### Download the Dataset

To download the dataset from Kaggle, use the following command:

```bash
kaggle datasets download -d stepandupliak/predict-poison-mushroom-by-photo
```

## Model Architecture

The model is based on the InceptionV3 architecture with the following modifications:

- **GlobalAveragePooling2D layer:** To reduce the dimensionality of the feature maps.
- **Dense layer with 1024 neurons and ReLU activation:** To learn complex patterns.
- **Dropout layer with a rate of 0.3:** To prevent overfitting.
- **Dense layer with 512 neurons and ReLU activation:** Additional layer for learning.
- **Dropout layer with a rate of 0.3:** Additional dropout for regularization.
- **Output layer with 1 neuron and sigmoid activation:** For binary classification.

## Model Training, Fine-Tuning, and Exporting

The model is trained using several techniques to improve its performance:

- **Data Augmentation:** Techniques such as rotation, width shift, height shift, shear, zoom, and horizontal flip are applied to the training images to make the model robust to various transformations.
- **Callbacks:** The following callbacks are used during training:
  - `ReduceLROnPlateau`: Reduces the learning rate when the validation loss plateaus.
  - `EarlyStopping`: Stops training early if the validation loss does not improve for a specified number of epochs.
  - `ModelCheckpoint`: Saves the best model based on the validation loss.

The model is initially trained with the base layers frozen, and then fine-tuned by unfreezing some of the top layers of the InceptionV3 model and retraining with a smaller learning rate to refine the model's weights.

After training and fine-tuning, the model is exported in the TensorFlow SavedModel format and converted to TensorFlow Lite format for deployment on mobile or embedded devices. The labels are also saved in a text file.

## Model Evaluation

The model's performance is evaluated on the validation set using various metrics:

- **Precision, Recall, and F1 Score:** These metrics provide insights into the model's accuracy, the proportion of actual positives correctly identified, and the balance between precision and recall, respectively.
- **Confusion Matrix:** A heatmap is generated to visualize the true positive, true negative, false positive, and false negative counts.
- **Classification Report:** A detailed report is generated to provide precision, recall, and F1 scores for each class.

## Usage

1. **Extract the Dataset:** Ensure the dataset is extracted to the appropriate directory.
2. **Load and Preprocess Dataset:** Use data augmentation techniques to preprocess the training images.
3. **Create and Compile the Model:** Construct the model using the InceptionV3 architecture and compile it with appropriate parameters.
4. **Train and Fine-Tune the Model:** Train the model using the training set and fine-tune it for better performance.
5. **Export the Model:** Convert the trained model to TensorFlow Lite format for deployment.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/EcemKonca/mushroom-classification.git
    cd mushroom-classification
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
