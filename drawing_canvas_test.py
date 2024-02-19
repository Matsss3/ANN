import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Drawing_Panel:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing Panel")
        
        self.canvas = Canvas(self.master, width=400, height=400, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.save_button = Button(self.master, text="Save", command=self.save_img)
        self.save_button.pack(side=tk.BOTTOM)
        
        self.old_x = None
        self.old_y = None
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        x, y = event.x, event.y
        r = 3  
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
        
    def save_img(self):
        temp_path = 'temp.png'
        self.image.save(temp_path)
        print(f'Updating image at: {temp_path}')

def main_drawing():
    root = tk.Tk()
    app = Drawing_Panel(root)
    root.mainloop()
    
main_drawing()

def pre_process_image(URL):
    image_data = cv2.imread(URL, cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28, 28))
    image_data = 255 - image_data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    
    return image_data
    
final_img = pre_process_image('temp.png')
print(final_img)