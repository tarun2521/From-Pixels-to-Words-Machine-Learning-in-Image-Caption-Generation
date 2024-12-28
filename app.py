import tkinter as tk
from tkinter import filedialog, Label, Button, Text, PhotoImage
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle

# Load VGG16 model for feature extraction
def load_vgg16_model():
    vgg_model = VGG16()  # Load pre-trained VGG16
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    return vgg_model

# Load models and tokenizer
vgg_model = load_vgg16_model()
lstm_model = load_model('best_model2.keras')

with open('model_data.pkl', 'rb') as f:
    data = pickle.load(f)
    tokenizer = data['tokenizer']
    max_length = data['max_length']

# Map index to word
def idx_to_word(index, tokenizer):
    return tokenizer.index_word.get(index, None)

# Predict caption
def predict_caption(model, features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Generate caption
def generate_caption(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize for VGG16 input
    image = np.expand_dims(np.array(image), axis=0)
    image = preprocess_input(image)  # Preprocess for VGG16
    features = vgg_model.predict(image, verbose=0)  # Extract features
    caption = predict_caption(lstm_model, features, tokenizer, max_length)
    return ' '.join(caption.split()[1:-1])  # Remove <start> and <end>

# GUI Application
class ImageCaptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning App")
        self.root.geometry("600x400")
        
        self.label = Label(self.root, text="Upload an Image to Generate Caption", font=("Arial", 16))
        self.label.pack(pady=20)
        
        self.upload_button = Button(self.root, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)
        
        self.image_label = Label(self.root)
        self.image_label.pack(pady=10)
        
        self.result_label = Label(self.root, text="Caption will appear here", font=("Arial", 14), wraplength=500, justify="center")
        self.result_label.pack(pady=20)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Display the uploaded image
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img)
            self.image_label.image = img
            
            # Generate caption
            caption = generate_caption(file_path)
            self.result_label.config(text=caption)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptionApp(root)
    root.mainloop()