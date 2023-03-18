import face_recognition



def recognize_face(image_path, known_faces_path):
    # Load the image and known faces
    image = face_recognition.load_image_file(image_path)
    known_faces = face_recognition.load_image_file(known_faces_path)

    # Find the face locations and encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load the known faces and names
    known_face_encodings = []
    known_face_names = []
    for face_image in known_faces:
        encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append('John Doe')

    # Iterate over the face encodings in the image and compare them to the known faces
    for face_encoding in face_encodings:
        # Check if the face encoding matches any of the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match is found, print the name of the person and return True
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print(f"Face recognized as {name}")
            return True

    # If no match is found, print an error message and return False
    print("Face not recognized")
    return False
