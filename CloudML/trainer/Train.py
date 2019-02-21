import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "C:/Users/Pedro/PycharmProjects/DiaRet/data/train-sample"
CATEGORIES = ["Healthy", "Disease"]

for category in CATEGORIES:
    path = os.path.join(DATADIR)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))

IMG_SIZE = 50

# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #Change to 3 when it's colour

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
X[1]



export JOB_NAME="test_job"
export BUCKET_NAME=kerastraindata
export CLOUD_CONFIG=trainer/cloudml-gpu.yaml
export JOB_DIR=gs://coinop-data/jobs/$JOB_NAME
export MODULE=trainer.cloud_trainer
export PACKAGE_PATH=./trainer
export REGION=europe-west2
export RUNTIME=1.2
export TRAIN_FILE=gs://kerastraindata/X.pickle

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version $RUNTIME \
    --module-name $MODULE \
    --package-path $PACKAGE_PATH \
    --region $REGION \
    --config-$CLOUD_CONFIG