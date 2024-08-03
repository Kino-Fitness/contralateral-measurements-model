import numpy as np
import pandas as pd
from PIL import Image
import torch
import ast
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import urllib.request
import pyheif
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import tensorflow.keras.backend as K
from ultralytics import YOLO
import numpy as np
from PIL import Image
   
def get_image(link):
    link_parts = link.split("/")
    file_id = link_parts[-2]
    parsed_link = f"https://drive.google.com/uc?id={file_id}"

    try:
        response = urllib.request.urlopen(parsed_link)
        
        heif_file = pyheif.read(response)

        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image = image.convert('RGB')
        model = YOLO("yolov8n.pt")
        results = model(image, classes=0)  # or specify custom classes
        boxes = results[0].boxes
        coords = boxes.xyxy.tolist()[0]
        image = image.crop(coords)
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_pil = Image.fromarray(image_array)
        return image_array

    except Exception as e:
        print(f"Error: {e}, link: {link}")


def str_to_tensor(tensor_str):
    nested_list = ast.literal_eval(tensor_str.strip())
    tensor = torch.tensor(nested_list)
    return tensor


def apply_random_augmentation(image):
    pil_image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=(-15, 15)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), 
        transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)), 
    ])

    transformed_image = transform(pil_image).convert('RGB')
    augmented_image = np.array(transformed_image)
    return augmented_image


class Scalar2Gaussian():
    def __init__(self,min=0.0,max=99.0,sigma=4.0,bins=10):
        self.min, self.max, self.bins, self.sigma = float(min), float(max), bins, sigma
        self.idxs = np.linspace(self.min,self.max,self.bins)
    def softmax(self, vector):
        e_x = np.exp(vector - np.max(vector))
        return e_x / e_x.sum()

    def code(self,scalar):
        probs = np.exp(-((self.idxs - scalar) / 2*self.sigma)**2)
        probs = probs/probs.sum()
        return probs
  
    def decode(self, vector):
        if np.abs(vector.sum()-1.0) < 1e-3 and np.all(vector>-1e-4):
            # print('Already Probability')
            probs=vector
        else: 
            probs = self.softmax(vector) #make sure vector is not already probabilities
        scalar = np.dot(probs, self.idxs)
        return scalar

    def decode_tensor(self, vector):
        def true_fn():
            return vector

        def false_fn():
            return tf.nn.softmax(vector)

        probs = tf.cond(
        tf.logical_and(
            tf.math.abs(tf.reduce_sum(vector) - 1.0) < 1e-3,
            tf.reduce_all(vector > -1e-4)
        ),
        true_fn,
        false_fn
        )

        scalar = tf.reduce_sum(probs * self.idxs)
        return scalar
  
s2g = {
    'right_bicep': Scalar2Gaussian(min=20.0, max=60.0),
    'left_bicep': Scalar2Gaussian(min=20.0, max=60.0),
    'chest': Scalar2Gaussian(min=60.0, max=170.0),
    'right_forearm': Scalar2Gaussian(min=15.0, max=40.0),
    'left_forearm': Scalar2Gaussian(min=15.0, max=40.0),
    'right_quad': Scalar2Gaussian(min=40.0, max=70.0),
    'left_quad': Scalar2Gaussian(min=40.0, max=70.0),
    'right_calf': Scalar2Gaussian(min=20.0, max=60.0),
    'left_calf': Scalar2Gaussian(min=20.0, max=60.0),
    'waist': Scalar2Gaussian(min=70.0, max=140.0),
    'hips': Scalar2Gaussian(min=80.0, max=110.0)
}

def process_data(df):
    X_front_images = []
    X_back_images = []
    X_tabular = []
    Y_right_bicep = []
    Y_left_bicep = []
    Y_chest = []
    Y_right_forearm = []
    Y_left_forearm = []
    Y_right_quad = []
    Y_left_quad = []
    Y_right_calf = []
    Y_left_calf = []
    Y_waist = []
    Y_hips = []
    Y_bodypose = []
    Y_joints = []

    for index, row in df.iterrows():
        X_front_images.append(row['Front Image'].astype(np.float32))
        X_back_images.append(row['Back Image'].astype(np.float32))
        X_tabular.append([row['Height '], row['Weight'], row['Demographic'], row['Gender']])
        Y_right_bicep.append(s2g['right_bicep'].code(row['Right Bicep']))
        Y_left_bicep.append(s2g['left_bicep'].code(row['Left Bicep']))
        Y_chest.append(s2g['chest'].code(row['Chest']))
        Y_right_forearm.append(s2g['right_forearm'].code(row['Right Forearm']))
        Y_left_forearm.append(s2g['left_forearm'].code(row['Left Forearm']))
        Y_right_quad.append(s2g['right_quad'].code(row['Right Quad']))
        Y_left_quad.append(s2g['left_quad'].code(row['Left Quad']))
        Y_right_calf.append(s2g['right_calf'].code(row['Right Calf']))
        Y_left_calf.append(s2g['left_calf'].code(row['Left Calf']))
        Y_waist.append(s2g['waist'].code(row['Waist']))
        Y_hips.append(s2g['hips'].code(row['Hips']))
        Y_bodypose.append(row['Body Pose']) 
        Y_joints.append(row['Joints'])   

    X_front_images = np.array(X_front_images)
    X_back_images = np.array(X_back_images)
    X_tabular = np.array(X_tabular)
    Y_right_bicep = np.array(Y_right_bicep)
    Y_left_bicep = np.array(Y_left_bicep)
    Y_chest = np.array(Y_chest)
    Y_right_forearm = np.array(Y_right_forearm)
    Y_left_forearm = np.array(Y_left_forearm)
    Y_right_quad = np.array(Y_right_quad)
    Y_left_quad = np.array(Y_left_quad)
    Y_right_calf = np.array(Y_right_calf)
    Y_left_calf = np.array(Y_left_calf)
    Y_waist = np.array(Y_waist)
    Y_hips = np.array(Y_hips)
    Y_bodypose = np.array(Y_bodypose)
    Y_joints = np.array(Y_joints)

    return X_front_images, X_back_images, X_tabular, Y_right_bicep, Y_left_bicep, Y_chest, Y_right_forearm, Y_left_forearm, Y_right_quad, Y_left_quad, Y_right_calf, Y_left_calf, Y_waist, Y_hips, Y_bodypose, Y_joints