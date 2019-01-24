
import sys
import face_recognition
import cv2
import os
from utils import create_csv

# Open the input movie file
input_movie = cv2.VideoCapture(sys.argv[1])
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("ester.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("bie.jpg")
al_face_encoding = face_recognition.face_encodings(al_image)[0]


known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

current_path = os.getcwd()

counbie = 0
counbie1 = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    if(frame_number % 8 == 0):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(
                known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match[0]:
                name = "ester"
            elif match[1]:
                name = "bie"

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left-22, top-42),
                          (right+22, bottom+32), (0, 0, 255), 2)

            crop_img = frame[top-40:bottom+30, left-20:right+20]
            if(name == "ester"):
                cv2.imwrite("./face_database/ester/" + "ester" + sys.argv[1] +
                            str(counbie)+".png", crop_img)
                counbie = counbie + 1

            elif(name == "bie"):
                cv2.imwrite("./face_database/bie/" + sys.argv[1] +
                            "bie"+str(counbie1)+".png", crop_img)
                counbie1 = counbie1 + 1

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left-10, bottom+10 - 35),
                          (right+10, bottom+10), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left-10 + 6, bottom+10 - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Write the resulting image to the output video file

        print("Writing frame {} / {}".format(frame_number, length))

        cv2.imshow('face_recog_crop', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# All done!
input_movie.release()
cv2.destroyAllWindows()
