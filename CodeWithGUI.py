import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("return/keras_model.h5", "return/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

root = tk.Tk()
root.title("Sign Language to Text Converter")

sentence = ""

def predict_sign():
    global sentence
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands, _ = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        sign = labels[index]
        sentence += sign
        label_var.set(sentence)

        cv2.rectangle(frame, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, sign, (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(frame, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_img.after(10, predict_sign)

def on_key_press(event):
    if event.char.lower() == 'k':
        predict_sign()

label_var = tk.StringVar()
label_var.set("")
label_output = tk.Label(root, textvariable=label_var, font=('Helvetica', 24))
label_output.pack()

label_img = tk.Label(root)
label_img.pack()

root.bind('<KeyPress>', on_key_press)

predict_sign()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
