import numpy as np
from keras_facenet import FaceNet

facenet = FaceNet()

def get_embedding(face_rgb):
    emb = facenet.embeddings([face_rgb])
    return emb[0]
