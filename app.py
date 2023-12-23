from threading import Thread
from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import os
import csv
from datetime import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Specify the path to the folder containing known faces
known_faces_folder = "faces"

# Load known faces and their names from the specified folder
known_faces = []
known_names = []

def reload_known_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []

    for filename in os.listdir("faces"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            face_image_path = os.path.join("faces", filename)

            # Load the image and convert to RGB format
            face_image = cv2.imread(face_image_path)
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Check if the image is grayscale and convert to RGB if necessary
            if len(face_image_rgb.shape) < 3 or face_image_rgb.shape[2] < 3:
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)

            # Encode the face and append to the list
            encoding = face_recognition.face_encodings(face_image_rgb)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

reload_known_faces()  # Initial loading of known faces

app.config['KNOWN_FACES'] = known_faces
app.config['KNOWN_NAMES'] = known_names

# Initialize attendance log
attendance_log = {}

def update_attendance(name):
    timestamp = datetime.now().strftime("%I:%M:%S %p")
    with open("attendance_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])

def gen_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        else:
            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Check if the face matches any known faces
                matches = face_recognition.compare_faces(app.config['KNOWN_FACES'], face_encoding)
                name = "Unknown"

                # Use the name of the first matching known face
                if True in matches:
                    first_match_index = matches.index(True)
                    name = app.config['KNOWN_NAMES'][first_match_index]

                    # Mark attendance for the recognized face
                    if name not in attendance_log or not attendance_log[name]:
                        attendance_log[name] = True
                        update_attendance(name)

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/home')
def homePage():
    return render_template('home.html')

# Registration of User here

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = known_faces_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'userImage' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['userImage']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Get the name from the form data
        name = request.form['name']

        # Securely save the file
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load the new face and add it to the known faces list
        new_face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        new_face_image = cv2.imread(new_face_image_path)
        new_face_image_rgb = cv2.cvtColor(new_face_image, cv2.COLOR_BGR2RGB)
        new_encoding = face_recognition.face_encodings(new_face_image_rgb)[0]

        app.config['KNOWN_FACES'].append(new_encoding)
        app.config['KNOWN_NAMES'].append(name)

        # Reload the known faces after a new user is registered
        reload_known_faces()

        # Optionally, you can store the name and filename in a database
        # For simplicity, we are just returning the name and filename in the response
        return jsonify({'name': name, 'filename': filename})

    return jsonify({'error': 'Invalid file format'})


if __name__ == '__main__':
    # Create or clear the attendance log CSV file
    with open("attendance_log.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])

    app.run(debug=True)
