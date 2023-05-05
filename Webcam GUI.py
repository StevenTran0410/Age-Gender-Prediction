import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from tkinter import *
import tkinter
import PIL.Image
import PIL.ImageTk
import time
from Utils import load_model, result

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

model = load_model('ArcFace')
print('Done')

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
str = ""

def get_bbox(results, image):
    for face_landmarks in results.multi_face_landmarks:
        # Extract relevant landmarks for the face bounding box
        points = []
        for id in range(0, 468):
            landmark = face_landmarks.landmark[id]
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            points.append([x, y])
        points = np.array([points], dtype=np.int32)
        
    # Draw the face bounding box
    bbox = cv2.boundingRect(points)
    return bbox

class App:
    global face_mesh

    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title = (window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(
            window, width=self.vid.width - 4, height=self.vid.height + 280)
        self.canvas.pack()

        text_frame = tkinter.Frame(
            window, background=self.from_rgb((17, 24, 31)))
        text_frame.place(x=0, y=self.vid.height+50, anchor="nw",
                         width=self.vid.width, height=208)

        self.text = tkinter.Text(
            text_frame, state='disable', width=115, height=19, font=("Times New Roman", 16), bg="#C0C0C0")
        self.text.pack()

        # scroll bar
        scrollbar = Scrollbar(self.text)
        scrollbar.place(relheight=1, relx=0.99)
        scrollbar.configure(command=self.text.yview)

        btn_frametop = tkinter.Frame(
            window, background=self.from_rgb((117, 123, 129)))
        btn_frametop.place(x=0, y=0, anchor="nw", width=self.vid.width)

        btn_framebot = tkinter.Frame(
            window, background=self.from_rgb((117, 123, 129)))
        btn_framebot.place(x=0, y=self.vid.height,
                           anchor="nw", width=self.vid.width)

        self.btn_results = tkinter.Button(
            btn_frametop, text="Detect", width=10, command=self.get_results, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_results.pack(side="left", padx=10, pady=10)
        self.window.bind("<space>", lambda event=None: self.get_results())

        self.btn_delete = tkinter.Button(
            btn_frametop, text="Delete All Text", width=15, command=self.delete_text, font=FONT_BOLD, bg=BG_GRAY)
        self.btn_delete.pack(side="right", padx=10, pady=10)

        self.delay = 15
        self.update()

        self.window.mainloop()
        
    def update_text(self):
        global str
        self.text.configure(state="normal")
        self.text.delete(1.0, END)
        self.text.insert('end', str)
        self.text.configure(state='disable')

    def delete_text(self):
        global str
        str = ''
        self.text.configure(state="normal")
        self.text.delete(1.0, END)
        self.text.configure(state='disable')

    def update(self):
        ret, frame = self.vid.get_frame(face_mesh)
        frame = cv2.resize(frame, (1280, 720))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.window.after(self.delay, self.update)

    def get_results(self):
        global results
        global cropped_image
        global str
        if results.multi_face_landmarks:
            age, gender = result(cropped_image, model)
            str += age.__str__()
            str += ' '
            str += gender.__str__()
            str += '\n'
            self.update_text()
        else:
            self.text.configure(state="normal")
            self.text.insert('end', '\n' + "No Faces Detect Yet")
            self.text.configure(state='disable')

    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb

class MyVideoCapture:
    """docstring for  MyVideoCapture"""

    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable open video source", video_source)
        
        self.width = 1280
        self.height = 720

    def get_frame(self, face_mesh):
        global results
        global cropped_image
        if self.vid.isOpened():
            ret, image = self.vid.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)    
            if results.multi_face_landmarks:
                x, y, w, h = get_bbox(results, image)
                z = int(5 * h / 100)
                v = int(2.5 * w / 100)
                cropped_image = image[y-z:y+h, x-v:x+w+v]
                cv2.rectangle(image, (x-v, y-z), (x+w+v, y+h), (0, 255, 0), 2)
            else:
                pass
            if ret:
                return (ret, image)
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        App(tkinter.Tk(), "tkinter ad OpenCV")
