import tkinter as tk
from PIL import ImageGrab
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

# Load updated model
model = tf.keras.models.load_model("digit_model.h5")
print("Model summary:")
model.summary() 

# GUI setup
root = tk.Tk()
root.title("Digit Recognizer")
canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.pack()
canvas.create_rectangle(1, 1, 199, 199, outline='red', width=1)
def test_with_mnist_sample():
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    sample = x_test[0]  # Should be a '7'
    sample = sample.reshape(1, 28, 28, 1).astype("float32") / 255.0
    pred = model.predict(sample)
    print("Test prediction on MNIST sample (should be 7):", np.argmax(pred))
    print("Full prediction probabilities:", pred)

# Run initial test
test_with_mnist_sample()

# [Rest of your existing GUI setup code...]

# Add the test button to your GUI (with other buttons)
test_button = tk.Button(root, text="Test Model", command=test_with_mnist_sample)
test_button.pack(pady=5)

def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x + 5, y + 5, fill='black', outline='black')

canvas.bind("<B1-Motion>", draw)

def preprocess_image(x, y, x1, y1):
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
    img = np.array(img)
    img = 255 - img  # Invert to white digit on black

    img = cv2.GaussianBlur(img, (5, 5), 0)  # Smooth edges
    
    # Use adaptive thresholding instead of fixed threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Find and extract the largest contour
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit = img[y:y+h, x:x+w]
        
        # Create blank image and place the digit in center
        processed = np.zeros_like(img)
        start_y = (processed.shape[0] - h) // 2
        start_x = (processed.shape[1] - w) // 2
        processed[start_y:start_y+h, start_x:start_x+w] = digit
        img = processed
    # else: use the original image if no contours found

    # Resize and pad
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Normalize
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)

def predict_digit():
    # Get canvas area on screen
    # Get the absolute screen coordinates of the canvas
    canvas_x = root.winfo_rootx() + canvas.winfo_x()
    canvas_y = root.winfo_rooty() + canvas.winfo_y()
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    # Calculate capture area with 1-pixel border adjustment
    x = canvas_x + 1
    y = canvas_y + 1
    x1 = canvas_x + canvas_width - 1
    y1 = canvas_y + canvas_height - 1
    
    print(f"Capturing canvas at: {x},{y} to {x1},{y1}")
    
    # Preprocess canvas image
    img = preprocess_image(x, y, x1, y1)

    # Optional: visualize what model sees
    import matplotlib.pyplot as plt
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title("Model input (preprocessed)")
    plt.axis('off')
    plt.show()

    # Predict using model
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show result
    result_label.config(text=f"Prediction: {digit} ({confidence:.2f}%)")




def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Draw a digit")

tk.Button(root, text="Predict", command=predict_digit).pack(pady=5)
tk.Button(root, text="Clear", command=clear_canvas).pack(pady=5)
result_label = tk.Label(root, text="Draw a digit", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
