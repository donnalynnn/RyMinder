import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Initialize the face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the JSON file
with open('data.json') as f:
    data = json.load(f)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Function to perform face recognition
def recognize_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (224, 224))
        face_image = transform(face_image).unsqueeze(0)
        output = model(face_image)
        _, predicted = torch.max(output.data, 1)
        label = data['labels'][predicted.item()]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return image

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_face(frame)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()