# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:38:22 2020

@author: Deep Chokshi
"""


import os
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

try:
    os.mkdir(r'D:/DataSets/data/AI/train_bread')    
    os.mkdir(r'D:/DataSets/data/AI/train_dairy_products')
    os.mkdir(r'D:/DataSets/data/AI/train_dessert')
    os.mkdir(r'D:/DataSets/data\AI/train_egg')
    os.mkdir(r'D:/DataSets/data\AI/train_fried_food')
    os.mkdir(r'D:/DataSets/data\AI/train_meat')
    os.mkdir(r'D:/DataSets/data\AI/train_noodles_pasta')
    os.mkdir(r'D:/DataSets/data\AI/train_rice')
    os.mkdir(r'D:/DataSets/data\AI/train_sea_food')
    os.mkdir(r'D:/DataSets/data\AI/train_soup')
    os.mkdir(r'D:/DataSets/data\AI/train_vegetables_fruit')
    os.mkdir(r'D:/DataSets/data\AI/test_bread')
    os.mkdir(r'D:/DataSets/data\AI/test_dairy_products')
    os.mkdir(r'D:/DataSets/data\AI/test_dessert')
    os.mkdir(r'D:/DataSets/data\AI/test_egg')
    os.mkdir(r'D:/DataSets/data\AI/test_fried_food')
    os.mkdir(r'D:/DataSets/data\AI/test_meat')
    os.mkdir(r'D:/DataSets/data\AI/test_noodles_pasta')
    os.mkdir(r'D:/DataSets/data\AI/test_rice')
    os.mkdir(r'D:/DataSets/data\AI/test_sea_food')
    os.mkdir(r'D:/DataSets/data\AI/test_soup')
    os.mkdir(r'D:/DataSets/data\AI/test_vegetables_fruit')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if (os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file size! i.e Zero length.')
    
    print(len(dataset))
    train_data_length = int(len(dataset) * SPLIT_SIZE)
    test_data_length = int(len(dataset) - train_data_length)
    #shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_data_length]
    test_set = dataset[-test_data_length:]
    
    for unitData in train_set:
        temp_train_data = SOURCE + unitData
        final_train_data = TRAINING + unitData
        copyfile(temp_train_data, final_train_data)
    
    for unitData in test_set:
        temp_test_data = SOURCE + unitData
        final_test_data = TESTING + unitData
        copyfile(temp_test_data, final_test_data)




Bread_SOURCE_DIR = r"D:/DataSets/data/train/Bread/"
TRAINING_Bread_DIR= r"D:/DataSets/data/AI/train_bread/"
TESTING_Bread_DIR = r"D:/DataSets/data/AI/test_bread/"
Dairy_SOURCE_DIR = "D:/DataSets/data/train/Dairy product/"
TRAINING_Dairy_DIR= r"D:/DataSets/data/AI/train_dairy_products/"
TESTING_Dairy_DIR = r"D:/DataSets/data/AI/test_dairy_Products/"
Dessert_SOURCE_DIR = r"D:/DataSets/data/train/Dessert/"
TRAINING_Dessert_DIR= r"D:/DataSets/data/AI/train_dessert/"
TESTING_Dessert_DIR = r"D:/DataSets/data/AI/test_dessert/"
Egg_SOURCE_DIR = r"D:/DataSets/data/train/Egg/"
TRAINING_Egg_DIR= r"D:/DataSets/data/AI/train_egg/"
TESTING_Egg_DIR = r"D:/DataSets/data/AI/test_egg/"
Fried_SOURCE_DIR = r"D:/DataSets/data/train/Fried Food/"
TRAINING_Fried_DIR= r"D:/DataSets/data/AI/train_fried_food/"
TESTING_Fried_DIR = r"D:/DataSets/data/AI/test_fried_food/"
Meat_SOURCE_DIR = r"D:/DataSets/data/train/Meat/"
TRAINING_Meat_DIR= r"D:/DataSets/data/AI/train_meat/"
TESTING_Meat_DIR = r"D:/DataSets/data/AI/test_meat/"
Noodles_SOURCE_DIR = r"D:/DataSets/data/train/Noodles-Pasta/"
TRAINING_Noodles_DIR= r"D:/DataSets/data/AI/train_noodles_pasta/"
TESTING_Noodles_DIR = r"D:/DataSets/data/AI/test_noodles_pasta/"
Rice_SOURCE_DIR = r"D:/DataSets/data/train/Rice/"
TRAINING_Rice_DIR= r"D:/DataSets/data/AI/train_rice/"
TESTING_Rice_DIR = r"D:/DataSets/data/AI/test_rice/"
Seafood_SOURCE_DIR = r"D:/DataSets/data/train/Seafood/"
TRAINING_Seafood_DIR= r"D:/DataSets/data/AI/train_sea_food/"
TESTING_Seafood_DIR = r"D:/DataSets/data/AI/test_sea_food/"
Soup_SOURCE_DIR = r"D:/DataSets/data/train\Soup/"
TRAINING_Soup_DIR= r"D:/DataSets/data/AI/train_soup/"
TESTING_sOUP_DIR = r"D:/DataSets/data/AI/test_soup/"
Veg_SOURCE_DIR = r"D:/DataSets/data/train/Vegetable-Fruit/"
TRAINING_Veg_DIR= r"D:/DataSets/data/AI/train_vegetables_fruit/"
TESTING_Veg_DIR = r"D:/DataSets/data/AI/test_vegetables_fruit/"


split_size = .7
split_data(Bread_SOURCE_DIR, TRAINING_Bread_DIR, TESTING_Bread_DIR, split_size)
split_data(Dairy_SOURCE_DIR, TRAINING_Dairy_DIR, TESTING_Dairy_DIR, split_size)
split_data(Dessert_SOURCE_DIR, TRAINING_Dessert_DIR, TESTING_Dessert_DIR, split_size)
split_data(Egg_SOURCE_DIR, TRAINING_Egg_DIR, TESTING_Egg_DIR, split_size)
split_data(Fried_SOURCE_DIR, TRAINING_Fried_DIR, TESTING_Fried_DIR, split_size)
split_data(Meat_SOURCE_DIR, TRAINING_Meat_DIR, TESTING_Meat_DIR, split_size)
split_data(Noodles_SOURCE_DIR, TRAINING_Noodles_DIR, TESTING_Noodles_DIR, split_size)
split_data(Rice_SOURCE_DIR, TRAINING_Rice_DIR, TESTING_Rice_DIR, split_size)
split_data(Seafood_SOURCE_DIR, TRAINING_Seafood_DIR, TESTING_Seafood_DIR, split_size)
split_data(Soup_SOURCE_DIR, TRAINING_Soup_DIR, TESTING_sOUP_DIR, split_size)
split_data(Veg_SOURCE_DIR, TRAINING_Veg_DIR, TESTING_Veg_DIR, split_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

TRAINING_DIR = "D:/DataSets/data/AI/training"
train_datagen = ImageDataGenerator(rescale=1.0/255)

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

VALIDATION_DIR = "D:/DataSets/data/AI/testing"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)