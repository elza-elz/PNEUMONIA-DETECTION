import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('model.keras')
    
def browse_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        display_image(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        prediction = model.predict(img_data)
        if prediction[0][0] > prediction[0][1]:
            result_label.config(text='Person is safe.', fg='green')
        else:
            result_label.config(text='Person is affected with Pneumonia.', fg='red')
        
def display_image(filepath):
    img = Image.open(filepath)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Create the main window
root = tk.Tk()
root.title("Pneumonia Detection")

# Add widgets
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
result_label = tk.Label(root, text="", font=("Helvetica", 14))
prediction_label = tk.Label(root, text="", font=("Helvetica", 12))
image_label = tk.Label(root)

# Grid layout
browse_button.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")  # Center the button
result_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
prediction_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
image_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

# Start the Tkinter event loop
root.mainloop()