"""Usage: install dependencies (see README), then run cells or execute as a script.
On Windows install dlib via conda-forge: conda install -c conda-forge dlib
"""

import sys
import subprocess
import os
import shutil
import random
import sys
import subprocess
import os
import shutil
import random

def _pip_install(packages):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
    except Exception as _e:
        print('Warning: pip install failed for', packages, '-', _e)


_pip_install(['--upgrade','pip'])
_pip_install(['opencv-python-headless==4.7.0.72','numpy','tqdm','imutils','tensorflow','keras','kagglehub'])

try:
    _pip_install(['dlib'])
except Exception:
    print('dlib install attempted and failed — if on Windows, consider: conda install -c conda-forge dlib')




OUT_DIR = 'kaggle_data/yawn_eye'
os.makedirs(OUT_DIR, exist_ok=True)
import os
OUT_DIR = 'kaggle_data/yawn_eye'
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import kagglehub
    print("kagglehub available — attempting download (may prompt for credentials).")
    kagglehub.login()
    handle = 'serenaraju/yawn-eye-dataset-new'
    kagglehub.dataset_download(handle, path=OUT_DIR, unzip=True, force_download=False)
    print("Downloaded with kagglehub into", OUT_DIR)
except Exception as e:
    print("kagglehub download failed or not available:", e)
    print("Try Kaggle CLI (you must place kaggle.json in ~/.kaggle/) — running fallback command if kaggle is installed.")
    try:
        subprocess.check_call(['kaggle','datasets','download','-d','serenaraju/yawn-eye-dataset-new','-p',OUT_DIR,'--unzip'])
        print("Downloaded with Kaggle CLI into", OUT_DIR)
    except Exception as e2:
        print("Kaggle CLI failed as well. Please download the dataset manually from Kaggle and place it under", OUT_DIR)


root = OUT_DIR
for root_dir, dirs, files in os.walk(root):
    print("DIR:", root_dir)
    print("Subdirs:", dirs)
    print("Sample files:", files[:10])
    break


src = OUT_DIR
dst_base = 'data'
train_open = os.path.join(dst_base,'train','open')
train_closed = os.path.join(dst_base,'train','closed')
val_open = os.path.join(dst_base,'val','open')
val_closed = os.path.join(dst_base,'val','closed')
for p in [train_open, train_closed, val_open, val_closed]:
    os.makedirs(p, exist_ok=True)

image_files = []
for root_dir, dirs, files in os.walk(src):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            image_files.append(os.path.join(root_dir,f))

print("Found", len(image_files), "images. Splitting into train/val...")

random.shuffle(image_files)
nval = max(1, int(0.2 * len(image_files)))

def label_from_name(fname):
    n = fname.lower()
    if 'open' in n: return 'open'
    if 'closed' in n or 'close' in n: return 'closed'
    d = os.path.basename(os.path.dirname(fname)).lower()
    if 'open' in d: return 'open'
    if 'closed' in d or 'close' in d: return 'closed'
    return 'open'

for i, fpath in enumerate(image_files):
    lab = label_from_name(fpath)
    if i < nval:
        dst = os.path.join(dst_base,'val',lab, os.path.basename(fpath))
    else:
        dst = os.path.join(dst_base,'train',lab, os.path.basename(fpath))
    shutil.copy(fpath, dst)

print("Copied. Train size:", sum(len(files) for _,_,files in os.walk(os.path.join(dst_base,'train'))),
      "Val size:", sum(len(files) for _,_,files in os.walk(os.path.join(dst_base,'val'))))


import cv2, dlib, os, numpy as np
from imutils import face_utils

PRED_PATH = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(PRED_PATH):
    print("Warning: dlib shape predictor not found at", PRED_PATH)
else:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PRED_PATH)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def crop_eye_from_image(img_path, out_path, resize=(64,64)):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return False
    rect = rects[0]
    shape = face_utils.shape_to_np(predictor(gray, rect))
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    def crop(eye):
        x1 = int(np.min(eye[:,0]) - 5)
        y1 = int(np.min(eye[:,1]) - 5)
        x2 = int(np.max(eye[:,0]) + 5)
        y2 = int(np.max(eye[:,1]) + 5)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = max(0,x2), max(0,y2)
        return cv2.resize(img[y1:y2, x1:x2], resize)
    le = crop(leftEye)
    re = crop(rightEye)
    cv2.imwrite(out_path.replace('.png','_L.png'), le)
    cv2.imwrite(out_path.replace('.png','_R.png'), re)
    return True

if os.path.exists(PRED_PATH):
    src_dir = 'data/train/open'
    dst_eye_dir = 'data_eyes/train/open'
    os.makedirs(dst_eye_dir, exist_ok=True)
    files = os.listdir(src_dir)[:200]
    cnt = 0
    for f in files:
        ok = crop_eye_from_image(os.path.join(src_dir,f), os.path.join(dst_eye_dir, f))
        if ok: cnt += 1
    print("Extracted", cnt, "eye crops to", dst_eye_dir)
else:
    print("Skipping eye extraction — download shape_predictor_68_face_landmarks.dat to enable this step.")


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (64,64)
batch = 32
train_dir = 'data/train'
val_dir = 'data/val'

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("Train/Val directories not found under 'data/'. Please prepare dataset first.")
else:
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1,
                                   horizontal_flip=True).flow_from_directory(train_dir, target_size=img_size, batch_size=batch, class_mode='binary')
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=img_size, batch_size=batch, class_mode='binary')

    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size[0],img_size[1],3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save('eye_model.h5')
    print("Saved model to eye_model.h5")


import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import time

PRED_PATH = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(PRED_PATH):
    print("Missing dlib predictor:", PRED_PATH)
else:
    EAR_THRESH = 0.25
    EAR_CONSEC_FRAMES = 20
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PRED_PATH)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    cap = cv2.VideoCapture(0)
    counter = 0
    alarm_on = False
    print("Starting EAR demo — press 'q' window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = face_utils.shape_to_np(predictor(gray, rect))
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftHull], -1, (0,255,0), 1)
            cv2.drawContours(frame, [rightHull], -1, (0,255,0), 1)
            if ear < EAR_THRESH:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                counter = 0
                alarm_on = False
        cv2.imshow("EAR Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


import cv2, dlib, os, numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model

MODEL_PATH = 'eye_model.h5'
PRED_PATH = 'shape_predictor_68_face_landmarks.dat'

if not os.path.exists(MODEL_PATH):
    print("Trained model not found:", MODEL_PATH)
elif not os.path.exists(PRED_PATH):
    print("dlib predictor not found:", PRED_PATH)
else:
    model = load_model(MODEL_PATH)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PRED_PATH)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def crop_eye(frame, eye_pts):
        x1 = int(np.min(eye_pts[:,0]) - 5)
        y1 = int(np.min(eye_pts[:,1]) - 5)
        x2 = int(np.max(eye_pts[:,0]) + 5)
        y2 = int(np.max(eye_pts[:,1]) + 5)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = max(0,x2), max(0,y2)
        crop = frame[y1:y2, x1:x2]
        try:
            crop = cv2.resize(crop, (64,64))
        except:
            return None
        return crop

    cap = cv2.VideoCapture(0)
    closed_count = 0
    THRESH_PROB = 0.6
    CONSEC_FRAMES = 20
    print("Starting CNN demo — press 'q' window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = face_utils.shape_to_np(predictor(gray, rect))
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            le = crop_eye(frame, leftEye)
            re = crop_eye(frame, rightEye)
            if le is None or re is None:
                continue
            p_left = model.predict(np.expand_dims(le.astype('float32')/255.0,0))[0][0]
            p_right = model.predict(np.expand_dims(re.astype('float32')/255.0,0))[0][0]
            closed_prob = 1.0 - ((p_left + p_right) / 2.0)
            if closed_prob > THRESH_PROB:
                closed_count += 1
                if closed_count >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                closed_count = 0
        cv2.imshow("CNN Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


print('Wrap-up: if dlib install fails on Windows use conda install -c conda-forge dlib; or consider MediaPipe as an alternative.')


