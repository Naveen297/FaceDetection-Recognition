import os
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import dlib

detector = dlib.get_frontal_face_detector()

def get_face_encodings_and_draw(image, draw_shape='rectangle'):
    image = image.convert('RGB')
    image_np = np.array(image)

    detected_faces = detector(image_np, 1)

    if len(detected_faces) == 0:
        return None, image

    draw = ImageDraw.Draw(image)
    face_encodings = []

    for face in detected_faces:
        if draw_shape == 'rectangle':
            draw.rectangle(((face.left(), face.top()), (face.right(), face.bottom())), outline=(255, 0, 0))
        elif draw_shape == 'circle':
            center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
            radius = max(face.right() - face.left(), face.bottom() - face.top()) // 2
            draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline=(255, 0, 0))

        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        face_location = (top, right, bottom, left)

        encoding = face_recognition.face_encodings(image_np, known_face_locations=[face_location])[0]
        face_encodings.append(encoding)

    return face_encodings[0], image

def save_image(image, path, filename):
    save_path = os.path.join(path, filename)
    image.save(save_path)

def main():
    # Hardcoded image paths
    uploaded_image_path = '1.jpg'
    captured_image_path = '3.jpg'

    # Directory to save new images
    output_dir = 'output_faces'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    uploaded_image = Image.open(uploaded_image_path)
    captured_image = Image.open(captured_image_path)

    uploaded_face_encoding, uploaded_image_with_rect = get_face_encodings_and_draw(uploaded_image)
    captured_face_encoding, captured_image_with_rect = get_face_encodings_and_draw(captured_image)

    if uploaded_face_encoding is None or captured_face_encoding is None:
        print("No face found in one or both images.")
        return

    results = face_recognition.compare_faces([uploaded_face_encoding], captured_face_encoding)
    print("Result: Pass, Face matched" if results[0] else "Result: Fail, Face Not matched")

    # Save images with rectangles
    save_image(uploaded_image_with_rect, output_dir, 'uploaded_with_rect.jpg')
    save_image(captured_image_with_rect, output_dir, 'captured_with_rect.jpg')

if __name__ == "__main__":
    main()
