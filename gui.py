import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import re
from tkinter import Tk, Frame, Label, StringVar, PhotoImage, Button
import customtkinter

class newproject:
    def __init__(self, master):
        self.window = master
        self.window.state('zoomed')  # Make the window fullscreen
        self.window.title('PROJECT')
        self.window.configure(bg='#f2f2f2')

        self.register_frame = Frame(self.window, bg='#c748a7', width=1440, height=960)
        self.register_frame.place(x=1, y=1)

        # Load the background image
        self.bg_image = PhotoImage(file='Gambar/bgpemro.png')

        # Use a Label to display the background image
        self.bg_label = Label(self.register_frame, image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.timpa = customtkinter.CTkFrame(self.register_frame, fg_color="#f2f2f2", corner_radius=5, width=444,
                                            height=60)
        self.timpa.place(x=804, y=623)

        self.entry = customtkinter.CTkEntry(master=self.register_frame,
                                            placeholder_text="Type your title here....",
                                            width=444,
                                            height=60,
                                            font=('Comic Sans MS', 22),
                                            bg_color="#c748a7",
                                            border_width=2,
                                            corner_radius=45)
        self.entry.place(x=804, y=347)

        # StringVar to store the prediction result
        self.prediction_result = StringVar()

        def predict():
            def lowercase(text):
                return text.lower()

            def remove_unnecessary_char(text):
                text = re.sub('\n', ' ', text)
                text = re.sub('rt', ' ', text)
                text = re.sub('user', ' ', text)
                text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
                text = re.sub('  +', ' ', text)
                text = re.sub('ï¿½', ' ', text)
                text = re.sub('ï¿', ' ', text)
                text = re.sub('ý', ' ', text)
                text = re.sub('ï', ' ', text)
                return text

            def remove_symbolnumeric(text):
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                return text

            def preprocess(text):
                text = lowercase(text)
                text = remove_unnecessary_char(text)
                text = remove_symbolnumeric(text)
                return text

            with open('vokabuler.pkl', 'rb') as file:
                vokabuler_loaded = pickle.load(file)

            vectorizer_baru = TfidfVectorizer(vocabulary=vokabuler_loaded)

            teks_baru = self.entry.get()
            teks_baru = preprocess(teks_baru)
            teks_baru_transformed = vectorizer_baru.fit_transform([teks_baru])

            svm_model = joblib.load('svm_tfidf.sav')
            prediksi = svm_model.predict(teks_baru_transformed)

            # Update the label with the prediction result
            self.pred.config(text=prediksi[0])

            # Print the prediction result in the terminal
            print("Prediction:", prediksi[0])

        # Load button image
        self.button_image_1 = PhotoImage(file="Gambar/button_1.png")

        # Create and display the button
        self.button_1 = Button(image=self.button_image_1, borderwidth=0, highlightthickness=0, command=lambda: predict(), relief="flat")
        self.button_1.image = self.button_image_1  # Keep a reference to avoid garbage collection
        self.button_1.place(x=873, y=490)

        # Create a label to display the prediction result
        self.pred = Label(self.register_frame, text="", bg="#c748a7", font=('Times New Roman', 40))
        self.pred.place(x=804, y=623)

window = Tk()
root = newproject(window)
window.mainloop()