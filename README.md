#🧠 MNIST-CNN-Recognizer

A Python-based GUI application to recognize hand-drawn digits (0-9) using a Convolutional Neural Network (CNN) trained on a real-world handwritten digit dataset. The user draws a digit with the mouse, and the model predicts it in real-time.

Built using:

🖌️ Tkinter for drawing interface

🔍 OpenCV for image preprocessing

🤖 TensorFlow/Keras for model prediction

📊 Dataset from Kaggle: https://www.kaggle.com/datasets/riyaldi/handwriting-digit-0-9

📸 Demo


🧮 Model Summary:

<img src="images/Capture5.PNG" alt="Model Summary" width="500"/>


✍️ Drawn Inputs and Predictions:

Drawn Input (GUI)	Prediction Output
<img src="images/Capture6.PNG" width="220"/>	
<img src="images/Capture7.PNG" width="220"/>
<img src="images/Capture9.PNG" width="220"/>

🗂️ Dataset Used
This project uses the Handwriting Digit 0–9 dataset from Kaggle.

Structure:

Copy
Edit
handDigitDataset/
├── 0/
├── 1/
├── ...
├── 9/
Each folder contains real handwritten digits in .jpg format across various styles and formats — ideal for training a robust digit recognition model.

🚀 How It Works
Draw a digit in the GUI canvas.

The drawn image is:

Captured using ImageGrab

Preprocessed with OpenCV (thresholding, resizing, centering)

Image is passed to the trained CNN model.

Prediction is shown with probability.

🏗️ Project Structure
bash
Copy
Edit
MNIST-CNN-Recognizer/
├── digit_gui.py           # GUI interface
├── cnn_model.h5           # Trained CNN model
├── train_model.py         # Model training script
├── images/                # Screenshots
└── README.md
🧠 Model Architecture
CNN Model Layers:

2× Conv2D + ReLU

2× MaxPooling2D

1× Dense (128) + Dropout

Final Softmax (10 outputs)

Training Settings:

Input shape: (28, 28, 1)

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 10

🖥️ Setup Instructions
Clone the repository:

bash
Copy
Edit
git clone https://github.com/FatimaRana50/MNIST-CNN-Recognizer.git
cd MNIST-CNN-Recognizer
Install dependencies:

bash
Copy
Edit
pip install tensorflow opencv-python pillow numpy matplotlib
Run the GUI:

bash
Copy
Edit
python digit_gui.py
📌 Future Improvements
Improve preprocessing pipeline for noisy drawings

Show confidence graph for all 10 digits

Allow live dataset collection from user drawings

Deploy to Streamlit or Flask for web use

🙌 Credits
Dataset: https://www.kaggle.com/datasets/riyaldi/handwriting-digit-0-9

Model & GUI: Built with Python, TensorFlow, Tkinter, and OpenCV



