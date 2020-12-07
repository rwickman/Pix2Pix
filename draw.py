# from tkinter import *
# from tkinter import ttk

# def savePosn(event):
#     global lastx, lasty
#     #print(event)
#     lastx, lasty = event.x, event.y

# def addLine(event):
#     print(event)
#     canvas.create_line((lastx, lasty, event.x, event.y))
#     savePosn(event)

# root = Tk()
# root.columnconfigure(0, weight=1)
# root.rowconfigure(0, weight=1)

# canvas = Canvas(root)
# canvas.grid(column=0, row=0, sticky=(N, W, E, S))
# canvas.bind("<Button-1>", savePosn)
# canvas.bind("<B1-Motion>", addLine)

# root.mainloop()

from tkinter import *
from tkinter import ttk
from PIL import Image
import io
import numpy as np
import cv2
from pix2pix import Pix2Pix, ImageDataLoader
import matplotlib.pyplot as plt

class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        #button.pack()
        self._pix = Pix2Pix(restore=True)
        self._generator = self._pix._generator
        self._data_loader = ImageDataLoader()
        self._data_loader.load_datasets("output_edges/train/edges.tfrecords", "output_edges/test/test_edges.tfrecords")
        
        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)

        
    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        self.create_line((self.lastx, self.lasty, event.x, event.y), width=5)
        self.save_posn(event)
    
    def save(self):
        ps = self.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        
        img_arr = np.array(img)
        print("img_arr.shape", img_arr.shape)
        img_arr = cv2.cvtColor(255 - img_arr, cv2.COLOR_BGR2RGB)
        print(img_arr)
                
        # gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # img_arr[thresh == 255] = 0.0
        # img_arr[thresh != 255] = 1.0
        img_arr = np.reshape(cv2.resize(img_arr, (256, 256)), (1, 256, 256, 3))
        # for img in self._data_loader.test_ds.take(1):
        #     out = self._generator(img[0], training=True)
        #fake_img = cv2.imread("fake/10097.png")
        #temp = cv2.Canny(fake_img, 256, 256)
        
        # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        # img_arr = np.reshape(cv2.resize(img_arr, (256, 256)), (1, 256, 256, 3))
        
        # Normalize image
        img_arr = (img_arr / 127.5) - 1
        
        #print(img_arr * 0.5 + 0.5)

        out = self._generator(img_arr, training=True)
        # print(out.shape)
        #plt.imshow(img_arr[0] * 0.5 + 0.5)
        #plt.show()
        plt.imshow(out[0]* 0.5 + 0.5)
        plt.axis("off")
        plt.show()
        print(img_arr.shape)
        #img.save('filename.jpg', 'jpeg')
        print("Hello world!")

    def clear(self):
        self.delete("all")

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

sketch = Sketchpad(root)
sketch.grid(column=0, row=0, sticky=(N, W, E, S))

button = Button(root, text="Save Drawing",command=sketch.save)
button.grid(column=1, row=1, sticky= E)

clear_button = Button(root, text="Clear",command=sketch.clear)
clear_button.grid(column=2, row=1, sticky= E)

root.mainloop()