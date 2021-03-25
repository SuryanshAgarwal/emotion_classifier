import warnings
warnings.filterwarnings("ignore")
import cv2
import sys
import numpy as np
from model import FacialExpressionModel

# take input path of image
imagePath = sys.argv[1]

# import cascade classifier from computer-vision library
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_faces_predict_emotions(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=3,
            minSize=(30, 30))
    
    print(f"Found {len(faces)} Faces!")
    i=0
    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(image, pred, (x, y), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
        i+=1
    
    return image

image = detect_faces_predict_emotions(imagePath)
output_img_name = imagePath.split('.')[0].split('/')[-1]
status = cv2.imwrite(f'output_img/{output_img_name}.jpg', image)
print(f"Image faces_detected written to filesystem: {status}")
sys.exit(0)