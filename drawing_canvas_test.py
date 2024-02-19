import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw

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
        
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, x, y, fill='black', width=2)
            
        self.old_x = x
        self.old_y = y
        
    def save_img(self):
        temp_path = 'temp.png'
        self.image.save(temp_path)
        
root = tk.Tk()
app = Drawing_Panel(root)
root.mainloop()