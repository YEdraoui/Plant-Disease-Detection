📌 Plant Disease Detection
A deep learning model for detecting plant diseases using image classification.

Plant Disease Detection Models

Overview

This repository contains deep learning models trained for detecting plant diseases using image classification techniques. The models are developed using EfficientNet architectures and are fine-tuned for optimal performance.

Models Included

EfficientNet-B3 (MixUp Augmentation) - plant_disease_efficientnet_b3_mixup_best.pth

EfficientNet (MixUp Augmentation) - plant_disease_efficientnet_mixup_best.pth

These models were trained on the PlantDoc Dataset, which contains images of various plant species affected by different diseases.

Training Process

The models were trained using PyTorch with the following key steps:

Data Preprocessing: Images were resized, normalized, and augmented using MixUp.

Model Architecture: EfficientNet-based deep learning models.

Optimization: Adam optimizer with learning rate scheduling.

Loss Function: Cross-entropy loss for multi-class classification.

Evaluation Metrics: Accuracy, Precision, Recall, and F1-score.


🌱 Overview
This project aims to identify plant diseases from leaf images using a convolutional neural network (CNN). The model is trained on the PlantDoc Dataset, which consists of various plant disease images.

📂 Dataset
The dataset is not included in this repository due to its size. You can download it from the official repository:

🔗 PlantDoc Dataset

After downloading, place the dataset in the data/ directory.

⚙️ Installation & Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/YEdraoui/Plant-Disease-Detection.git
cd Plant-Disease-Detection
Create a virtual environment and activate it:
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
🚀 Usage
Run the model training script:
bash
Copy
Edit
python main.py
Or open main.ipynb in Jupyter Notebook for step-by-step execution.
🛠 Technologies Used
Python
TensorFlow/Keras
OpenCV
NumPy & Pandas
Matplotlib
⚖️ License
This project is open-source under the MIT License.
