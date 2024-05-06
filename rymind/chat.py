import random
import json
import speech_recognition as sr
import torch
from transformers import pipeline
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pyttsx3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ryry"
print("Let's chat! (type 'quit' to exit)")

# for TTS and STT 
nlp_pipeline = pipeline("text-generation", model="gpt2") 

def listen():
    # Initialize speech recognition
    recognizer = sr.Recognizer()
    while True:
    
        with sr.Microphone(device_index=2) as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
            response = generate_response(text)
            print(response)
        except sr.UnknownValueError:
            print(f"{bot_name}: Sorry, I didn't get that.")
            text_to_speech("Sorry, I didn't get that.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

engine = pyttsx3.init()
        
def text_to_speech(text):
    # Set the rate of speech
    engine.setProperty('rate', 150) # Speed of speech
    # Set the volume
    engine.setProperty('volume', 0.9) # Volume, 0.0 to 1.0
    # Convert the text to speech
    engine.say(text)
    # Wait for the speech to finish
    engine.runAndWait()
           
        
while True:
    
    sentence = listen()
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                text_to_speech(random.choice(intent['responses']))
    else:
        print(f"{bot_name}: I do not understand...")
        text_to_speech("I do not understand")
        



# INSTALLATIONS
# pip install google-cloud-speech
# pip install numpy torch nltk