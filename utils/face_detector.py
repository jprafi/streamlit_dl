import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_and_crop(image):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None

    largest = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = largest['box']
    margin = int(0.2 * max(w, h))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))

    return face
