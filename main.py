import face_recognition
import numpy as np
from os import system, listdir, path
import cv2

def say(text):
    system("gtts-cli '" + text + "' -t co.uk -o voice.mp3")
    system('ffplay voice.mp3 -autoexit')

clear = lambda: system('clear')
camera = cv2.VideoCapture(int(input("Camera ID> ")))

faces = []
faceNames = []

# say("Good afternoon Dobbles family, please standby while I get ready...")

for face in listdir('./faces'):
    faceFile = face_recognition.load_image_file('faces/' + face)
    faceEncoding = face_recognition.face_encodings(faceFile)[0]
    
    faceNames.append(path.splitext(face)[0])
    faces.append(faceEncoding)

pastName = ''

try:
    clear()
    say("Fully loaded!")

    while True:
        ret, frame = camera.read()

        smallFrame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
        rgbFrame = smallFrame[:, :, ::-1]

        print("Encoding image...")
        unknownLocation = face_recognition.face_locations(rgbFrame, model="mtcn")
        unknownEncoding = face_recognition.face_encodings(rgbFrame, unknownLocation)
        print("Finished encoding image...")

        if len(unknownEncoding) > 0:
            print("I see a face!")

            unknownEncoding = unknownEncoding[0]

            print('Comparing faces...')
            matches = face_recognition.compare_faces(faces, unknownEncoding, tolerance=0.5)
            print('Finished comparing faces...')

            name = 'unknown'        
            
            face_distances = face_recognition.face_distance(faces, unknownEncoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faceNames[best_match_index]

            if name != pastName:
                greeting = f'Hello {name}!'
            
                print(greeting)
                say(greeting) 
                
                pastName = name
        
        print("Next frame...")

except KeyboardInterrupt:
    print("Exiting gracefully...")
    camera.release()
    cv2.destroyAllWindows()
