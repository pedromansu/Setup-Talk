import cv2
import sys
from _datetime import datetime

def classificacao(image,faceCascade,eyeCascade):
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    olhos = eyeCascade.detectMultiScale(
        gray,
        minNeighbors=20,
        minSize=(10, 10),
        maxSize=(90,90)
    )


    print(("Found {0} Faces!".format(len(faces))))
    print(("Found {0} Olhos!".format(len(olhos))))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in olhos:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
    cv2.imshow("Faces and eyes found", image)
    cv2.waitKey(0)

###########################################################################################################################

    
# Get user supplied values
imagePath = sys.argv[1]
cascPath = "C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\haarcascade_frontalface_default.xml"
cascPath2 = "C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascPath2)

# Read the image
image = cv2.imread("C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\face.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Image("C:\\Users\\pedro.pereira\\Desktop\\Trabalho\\SetupTalk\\Computer Vision\\face.jpg")
classificacao(image,faceCascade,eyeCascade)