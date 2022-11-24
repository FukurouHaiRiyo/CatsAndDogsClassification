import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model = load_model('model.h5')

classes = {
      0: 'dog',
      1: 'cat',
}


top = tk.Tk()
top.geometry('600x600')
top.title('CatDogClassifier')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
      img = image.load_img(file_path, target_size = (64, 64))
      img = image.img_to_array(img)
      img = np.expand_dims(img, axis = 0)
      result = model.predict(img)

      

      if result[0][0] == 1:
        print('dog')
      else:
        print('cat')

def show_classify_button(file_path):
      classifyButton = Button(top, text='Classify', command=lambda:classify(file_path), padx=5, pady=5)
      classifyButton.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
      classifyButton.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="CatsVSDogs Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
