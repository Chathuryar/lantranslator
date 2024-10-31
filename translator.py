# offline_translation_tts.py

from tkinter import *
from tkinter import ttk
import pyttsx3  # Text-to-speech library
from transformers import MarianMTModel, MarianTokenizer
import os
from vosk import Model as VoskModel, KaldiRecognizer
import pyaudio  # Library for capturing live microphone input
import json

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained translation model and tokenizer from local files
def load_model_and_tokenizer(local_model_dir):
    tokenizer = MarianTokenizer.from_pretrained(local_model_dir)
    model = MarianMTModel.from_pretrained(local_model_dir)
    return model, tokenizer

# Specify local paths for the downloaded models
local_model_paths = {
    "English to Spanish": "./offline_models/english_to_spanish",
    "English to French": "./offline_models/english_to_french",
}

# Load Vosk model for offline speech recognition
# Update this path to where your Vosk model is saved
vosk_model_path = "./vosk-model-small-en-us-0.15"  # Replace with your Vosk model path

# Check if the Vosk model exists
if not os.path.exists(vosk_model_path):
    print(f"Vosk model not found at {vosk_model_path}")
    print("Please download a Vosk model and update the 'vosk_model_path' variable.")
    exit(1)

# Initialize Vosk model
vosk_model = VoskModel(vosk_model_path)

# Set up the GUI window
root = Tk()
root.geometry('1100x400')
root.resizable(0, 0)
root['bg'] = "skyblue"
root.title('Offline Text-to-Speech (TTS) with Offline Translation and Voice Input')

# Attempt to set the icon, if it exists
try:
    root.iconbitmap('logo simpli.ico')
except Exception as e:
    print(f"Warning: Icon file not found. Using default icon. Error: {e}")

# Create frames for better organization
frame_header = Frame(root, bg='skyblue')
frame_header.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky='ew')

frame_input = Frame(root, bg='skyblue')
frame_input.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

frame_output = Frame(root, bg='skyblue')
frame_output.grid(row=1, column=2, columnspan=2, padx=10, pady=10, sticky='ew')

frame_buttons = Frame(root, bg='skyblue')
frame_buttons.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky='ew')

# Header Label
Label(frame_header, text="Offline TTS with Offline Translation and Voice Input", font="Arial 20 bold", bg='skyblue').pack()

# Input Text
Label(frame_input, text="Enter Text or Use Voice Input", font='Arial 13 bold', bg='white smoke').grid(row=0, column=0, padx=5, pady=5, sticky='w')
Input_text = Entry(frame_input, width=60)
Input_text.grid(row=1, column=0, padx=5, pady=5)

# Output Text
Label(frame_output, text="Output (translated text)", font='Arial 13 bold', bg='white smoke').grid(row=0, column=0, padx=5, pady=5, sticky='w')
Output_text = Text(frame_output, font='Arial 10', height=5, wrap=WORD, padx=5, pady=5, width=50)
Output_text.grid(row=1, column=0, padx=5, pady=5)

# Create a list of languages for the dropdown
available_languages = ["English to Spanish", "English to French"]

# Create a dropdown with available languages
dest_lang = ttk.Combobox(frame_input, values=available_languages, width=40)
dest_lang.grid(row=2, column=0, padx=5, pady=5)
dest_lang.set('Choose Translation')

# Offline translation functionality
def TranslateText():
    input_text = Input_text.get()
    selected_language = dest_lang.get()

    if selected_language in local_model_paths:
        local_model_dir = local_model_paths[selected_language]
        model, tokenizer = load_model_and_tokenizer(local_model_dir)
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        translated = model.generate(inputs, max_length=100)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        Output_text.delete(1.0, END)
        Output_text.insert(END, translated_text)
    else:
        Output_text.delete(1.0, END)
        Output_text.insert(END, input_text)

# Text-to-speech functionality
def TextToSpeech():
    text = Output_text.get(1.0, END).strip()
    if text:
        engine.say(text)
        engine.runAndWait()

# Offline voice input functionality using Vosk
def VoiceInput():
    recognizer = KaldiRecognizer(vosk_model, 16000)
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    print("Listening for voice input...")

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            recognized_data = json.loads(result)
            input_text = recognized_data.get("text", "")
            
            # Update the input text field with the recognized voice input
            Input_text.delete(0, END)
            Input_text.insert(END, input_text)
            print(f"Voice input recognized: {input_text}")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

# Buttons for translating text, performing TTS, and voice input
trans_btn = Button(frame_buttons, text='Translate Text', font='Arial 12 bold', pady=5, command=TranslateText, bg='orange', activebackground='green')
trans_btn.grid(row=0, column=0, padx=10, pady=10)

tts_btn = Button(frame_buttons, text='Speak', font='Arial 12 bold', pady=5, command=TextToSpeech, bg='blue', activebackground='yellow')
tts_btn.grid(row=0, column=1, padx=10, pady=10)

voice_btn = Button(frame_buttons, text='Voice Input', font='Arial 12 bold', pady=5, command=VoiceInput, bg='green', activebackground='red')
voice_btn.grid(row=0, column=2, padx=10, pady=10)

root.mainloop()