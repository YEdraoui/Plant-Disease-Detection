📌 Plant Disease Detection
A deep learning model for detecting plant diseases using image classification.

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
