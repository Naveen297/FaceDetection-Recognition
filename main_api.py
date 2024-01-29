from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import dlib

app = Flask(__name__)
CORS(app)

detector = dlib.get_frontal_face_detector()

def get_face_encodings_and_draw(image):
    image = image.convert('RGB')
    image_np = np.array(image)

    detected_faces = detector(image_np, 1)

    if len(detected_faces) == 0:
        return None, image

    draw = ImageDraw.Draw(image)
    face_encodings = []

    for face in detected_faces:
        draw.rectangle(((face.left(), face.top()), (face.right(), face.bottom())), outline=(255, 0, 0))

        # Convert dlib's rectangle to a plain tuple in (top, right, bottom, left) format
        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        face_location = (top, right, bottom, left)

        # Get the face encoding and append to list
        encoding = face_recognition.face_encodings(image_np, known_face_locations=[face_location])[0]
        face_encodings.append(encoding)

    # Assuming the first face encoding is what you want to return
    return face_encodings[0], image

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    data = request.json
    uploaded_image_data = data['uploadedImage']
    captured_image_data = data['capturedImage']

    uploaded_image = Image.open(BytesIO(base64.b64decode(uploaded_image_data.split(',')[1])))
    captured_image = Image.open(BytesIO(base64.b64decode(captured_image_data.split(',')[1])))

    uploaded_face_encoding, uploaded_image_with_rect = get_face_encodings_and_draw(uploaded_image)
    captured_face_encoding, captured_image_with_rect = get_face_encodings_and_draw(captured_image)

    if uploaded_face_encoding is None or captured_face_encoding is None:
        return jsonify({"result": "No face found in one or both images."})

    results = face_recognition.compare_faces([uploaded_face_encoding], captured_face_encoding)

    result = {
        "result": "Pass" if results[0] else "Fail",
        "uploadedImage": encode_image_to_base64(uploaded_image_with_rect),
        "capturedImage": encode_image_to_base64(captured_image_with_rect)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
