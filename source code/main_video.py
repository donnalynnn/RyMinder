import cv2
import speech_recognition as sr
from transformers import pipeline
from simple_facerec import SimpleFacerec
import os
import shutil

# Initialize your face recognition system
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Assuming you have a dictionary of face details
face_details = {
    "Elon Musk": "Elon Musk is a software engineer.",
    "Jane": "Jane is a data scientist."
}

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize NLP pipeline for text generation
nlp_pipeline = pipeline("text-generation", model="gpt2")

def listen_and_respond():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        response = generate_response(text)
        print(response)
    except Exception as e:
        print("Sorry, I didn't get that.")

def generate_response(text):
    # Check if the user asked who the person is
    if "who is the person" in text.lower():
        # Capture an image from the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        # Use face recognition to identify the person
        face_locations, face_names = sfr.detect_known_faces(frame)
        if face_names:
            # Assuming the first recognized face is the one we're interested in
            name = face_names[0]
            details = face_details.get(name, "I don't have details about this person.")
            return f"The person is {name}. {details}"
        else:
            return "I couldn't recognize the person."
    else:
        # For other queries, just echo back the text
        return text
    
def save_person_info():
    name = input("Enter the name of the person: ")
    details = input("Enter details about the person: ")
    photo_path = input("Enter the path to the photo of the person: ")
    
    # Save the photo to the images directory
    new_photo_path = os.path.join("images", f"{name}.jpg")
    shutil.copy(photo_path, new_photo_path)
    
    # Update face_details with the new person's information
    face_details[name] = details
    
    # Reload face encodings to include the new photo
    sfr.load_encoding_images("images/")
    
    print(f"Information about {name} has been saved and added to the face recognition system.")

def continuous_video_capture_and_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)

        listen_and_respond()

        key = cv2.waitKey(1)
        if key == 27: # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("Menu:")
        print("1. Start detecting and recognizing person")
        print("2. Input info about a person")
        choice = input("Enter your choice: ")

        if choice == "1":
            continuous_video_capture_and_recognition()
        elif choice == "2":
            save_person_info()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
