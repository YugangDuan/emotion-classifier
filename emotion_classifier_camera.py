#-*- coding: utf-8 -*-

import cv2
import sys
import gc
import json
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)
root_path = './pic/'
model_path = root_path + '/model/'
img_size = 48
# emo_labels = ['angry','fear','happy','sad','surprise','neutral']
# load json and create model arch
emo_labels = ['angry', 'disgust:', 'fear',
              'happy', 'sad', 'surprise', 'neutral']
num_class = len(emo_labels)
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weight
model.load_weights(model_path + 'model_weight.h5')

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)

    # The color of the rectangle that frames the face
    color = (0, 0, 2555)

    # Captures live video streams from the specified camera
    cap = cv2.VideoCapture(0)

    # Face recognition classifier local storage path
    cascade_path = root_path + "haarcascade_frontalface_alt.xml"

    # Loop detection recognizes faces
    while True:
        _, frame = cap.read()  # Read a frame of video

        # The image is grayed to reduce the computational complexity
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using the face recognition classifier, read in the classifier
        cascade = cv2.CascadeClassifier(cascade_path)

        # The classifier is used to identify which region is the face
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.1,
                                             minNeighbors=1, minSize=(120, 120))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                images = []
                rs_sum = np.array([0.0] * num_class)
                # capture the face images and submitted to the model
                image = frame_gray[y: y + h, x: x + w]
                image = cv2.resize(image, (img_size, img_size))
                image = image * (1. / 255)
                images.append(image)
                images.append(cv2.flip(image, 1))
                images.append(cv2.resize(image[2:45, :], (img_size, img_size)))
                for img in images:
                    image = img.reshape(1, img_size, img_size, 1)
                    list_of_list = model.predict_proba(
                        image, batch_size=32, verbose=1)  # predict
                    result = [prob for lst in list_of_list for prob in lst]
                    rs_sum += np.array(result)
                print(rs_sum)
                label = np.argmax(rs_sum)
                emo = emo_labels[label]
                print('Emotion : ', emo)
                cv2.rectangle(frame, (x - 10, y - 10),
                              (x + w + 10, y + h + 10), color, thickness=2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # The tooltip
                cv2.putText(frame, '%s' % emo, (x + 30, y + 30),
                            font, 1, (255, 0, 255), 4)
        cv2.imshow("Identify expression", frame)

        # Wait for 10 milliseconds to see if a key is entered
        k = cv2.waitKey(30)
        # If you enter q, you exit the loop
        if k & 0xFF == ord('q'):
            break

    # Release the camera and destroy all Windows
    cap.release()
    cv2.destroyAllWindows()
