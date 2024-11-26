
# Dr. Derma: Leveraging Deep Learning for Reliable Skin Lesion Classification

## Overview

Skin cancer is one of the most prevalent cancers globally, with diagnostic challenges stemming from the variability in lesion appearance and a shortage of dermatologists. **Dr. Derma** is a deep learning-based solution designed to improve the accuracy and reliability of automated skin cancer detection. By leveraging the VGG19 architecture and advanced data augmentation techniques, our model performs multi-class classification on skin lesions, offering a robust tool for aiding medical professionals.

---

## Datasets

Our project utilizes two primary datasets:

1. **[ISIC-9 Dataset](https://www.isic-archive.com/)**: 
   - **Manual Download Required**: 
     - Download the dataset.
     - Upload it to your Google Drive.
     - Provide the dataset's path in the code.
   - Contains 2,357 images categorized into 9 skin lesion types.
   - Images are pre-split into training (2,213 images) and testing (144 images) sets.

2. **[HAM10000 Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic?resource=download)**:
   - **Direct Import**: This dataset is fetched directly within the code.
   - Comprises 10,000 images representing 7 skin lesion types.
   - Used as is, with no manual configuration required.

By combining these datasets, we expanded the training dataset to 15,298 images, addressing class imbalance and improving model performance.

---

## Model Architecture

Our model leverages the **VGG19 architecture**, which processes images through:

- **Conv2D Layers**: Extract features such as edges, textures, and patterns.
- **ReLU Activation**: Introduces non-linearity for learning complex patterns.
- **MaxPooling**: Reduces spatial dimensions while retaining critical features.
- **Dense Layers**: Learn relationships between features to output class probabilities.
- **Softmax Activation**: Produces probabilities for multi-class classification.

Images are normalized to **224x224 pixels** before input, and data augmentation techniques—rotation, contrast adjustments, and zoom—are applied to enhance generalizability.

---

## Key Results

### Performance Metrics
| Dataset          | Test Accuracy | Loss  | Macro F1 Score |
|-------------------|---------------|-------|----------------|
| ISIC-9           | 51.56%        | 1.7688| 77.16%         |
| ISIC-9 + HAM10000| 73.75%        | 1.160 | 62.6%          |

Combining datasets resulted in a **22.19% increase in test accuracy** and significantly reduced model loss.

---

## Setup and Usage

### Requirements

- Python 3.11.7
- Required libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib
- Google Colab or local machine with GPU for training

### Steps to Run

1. Clone this repository.
2. Download the **ISIC-9 dataset** from [here](https://www.isic-archive.com/) and upload it to your Google Drive.
   - Provide the path to the dataset in the configuration file or notebook.
3. The **HAM10000 dataset** will be automatically imported during runtime.
4. Run the training script to train the model.
5. Evaluate the model on the test set to obtain classification metrics.

---

## Visualizations

Our analysis includes:

- **Dataset Distribution**: Bar plots showing the count of images in each lesion category for training, validation, and test sets.
- **Confusion Matrix**: Visualization of classification performance for individual lesion types.

---

## Conclusion

**Dr. Derma** demonstrates state-of-the-art performance in automated skin lesion classification, emphasizing the value of diverse datasets and data augmentation. The model sets a foundation for accessible diagnostic tools, aiding healthcare professionals in timely and accurate skin cancer detection.

---

