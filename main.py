import face_recognition
import os
import cv2

KNOWN_FACES_DIRECTORY = 'users'
STRANGERS_DIRECTORY = 'strangers'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'





print("loading users")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIRECTORY):
    for pic in os.listdir(f'{KNOWN_FACES_DIRECTORY}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIRECTORY}/{name}/{pic}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding) # add the face of the existing user to the list
        known_names.append(name) #add the name of the existing user to the list 
print("Succesfully registered users")


print("Reading unknown faces ...")
for pic in os.listdir(STRANGERS_DIRECTORY):
    print("Loading next person ...")
    image = face_recognition.load_image_file(f'{STRANGERS_DIRECTORY}/{pic}')
    locations = face_recognition.face_locations(image, model = MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print("Trying to see if the person is a stranger ...")
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match  = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            #draw a rectangle around the face
            color = [0, 255, 0]

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            #name tag
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
        else:
            print("this person is a stranger")
            #draw a rectangle around the face
            color = [255, 0, 0]

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            #name tag
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            
            cv2.putText(image, "Stranger!", (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow(pic, image)
    cv2.waitKey(3000)
#    cv2.destroyWindow(pic)



