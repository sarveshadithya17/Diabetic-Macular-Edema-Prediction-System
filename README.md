# Diabetic Macular Edema (DME) Prediction System

## Overview
This project aims to develop a deep learning-based system for detecting **Diabetic Macular Edema (DME)** from Optical Coherence Tomography (OCT) scans. The model is built using **U-Net architecture** for segmentation and is implemented using **TensorFlow, OpenCV, and SciPy**.

## Features
- **Deep Learning-based Image Segmentation** using **U-Net**.
- **Automatic DME detection** from OCT scans.
- **Google Colab TPU Support** for faster training.
- **Performance Metrics:** Accuracy, Dice Coefficient, IoU, and F1-score.
- **Data Augmentation** for better generalization.

## Technologies Used
- **Python** (TensorFlow, OpenCV, NumPy, SciPy, Matplotlib)
- **Flask** (for Web Application Integration)
- **Google Colab** (for Training on TPU/GPU)

## Installation
### Clone the Repository
```bash
$ git clone https://github.com/your-username/dme-prediction.git
$ cd dme-prediction
```
### Install Dependencies
```bash
$ pip install -r requirements.txt
```

## Training the Model
Run the following command to start training the model:
```bash
$ python train.py
```

## Running the Web Application
After training, you can run the Flask-based web interface using:
```bash
$ python app.py
```
The application will be available at: **http://127.0.0.1:5000/**

## Dataset
- The dataset consists of **OCT scans** with labeled masks for DME detection.
- You can use public datasets like **Duke OCT Dataset** or any custom dataset.

## Model Architecture
The U-Net model consists of:
- **Contracting Path:** Convolutional layers for feature extraction.
- **Expanding Path:** Transposed convolutions for upsampling.
- **Skip Connections:** To retain spatial information.

## Results
- Achieved **99.48% accuracy** on validation data.
- Dice Coefficient: **84%**

## Future Improvements
- Fine-tune using **pretrained encoders (EfficientNet, ResNet)**.
- Improve **data augmentation** for better robustness.
- Deploy on **Cloud (AWS/GCP)** for real-world usage.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes and push to GitHub.
4. Open a Pull Request.

## License
This project is licensed under the **MIT License**.

## Contact
For any queries, feel free to reach out:
- Email: **sarveshadithya.j.gmail@example.com**

