from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
# import cv2
# import os
# import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# HYPERPARAMETERS
BATCH_SIZE = 16
PATH_TO_TRAIN_DATA = "Data/training_set/"
PATH_TO_TEST_DATA = "Data/testing_set/"


# PREPARE TRAIN AND TEST SETS
train = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess_input, horizontal_flip=True)
test = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess_input)

train_set = train.flow_from_directory(PATH_TO_TRAIN_DATA,
                                      target_size=(224, 224),
                                      class_mode='categorical',
                                      batch_size=BATCH_SIZE,
                                      # save_format='png'
                                      )
test_set = test.flow_from_directory(PATH_TO_TEST_DATA,
                                    target_size=(224, 224),
                                    class_mode='categorical',
                                    batch_size=BATCH_SIZE,
                                    # save_format='png'
                                    )
X_train, y_train = train_set.next()
X_test, y_test = test_set.next()
print("Data preprocessing completed...")


# FUNCTION FOR CREATING A CNN MODEL
def build_model():
    """model from Kaggle https://www.kaggle.com/bradwel/facial-classification-model"""
    input_img = Input(shape=(224, 224, 3), name='ImageInput')
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(4, (3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    # x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='Conv2_1')(x)
    # x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='Conv2_2')(x)
    # x = MaxPooling2D((2, 2), name='pool2')(x)
    #
    # x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='Conv3_1')(x)
    # x = BatchNormalization(name='bn1')(x)
    # # x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv3_2')(x)
    # #x = BatchNormalization(name='bn2')(x)
    # x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    # x = MaxPooling2D((2, 2), name='pool3')(x)
    #
    # x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv4_1')(x)
    # x = BatchNormalization(name='bn3')(x)
    # x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv4_2')(x)
    # x = BatchNormalization(name='bn4')(x)
    # x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv4_3')(x)
    # x = MaxPooling2D((2, 2), name='pool4')(x)

    x = Flatten(name='flatten')(x)
    # x = Dense(1024, activation='relu', name='fc1')(x)
    # x = Dropout(0.7, name='dropout1')(x)
    x = Dense(8, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(5, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model


# Define model
ConvNet_model = build_model()
print(ConvNet_model.summary())

# Compile model
adam_optimizer = Adam(learning_rate=0.0001, decay=1e-5)
# es = EarlyStopping(patience=5, min_delta=0.05, monitor="val_accuracy")
#chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
ConvNet_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_optimizer)

# Fit model
history = ConvNet_model.fit(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    # steps_per_epoch=train_set.samples//BATCH_SIZE,
    # validation_split=0.2,
    validation_data=(X_test, y_test),
    # validation_steps=(train_set.samples / 5) // BATCH_SIZE,
    epochs=5
    # callbacks=[es, chkpt]
)

# RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()


plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.legend()
plt.show()

# SAVE MODEL AND WEIGHTS
ConvNet_model.save("TrainingModels/models_h5/CNN_2.h5")
ConvNet_model.save_weights("TrainingModels/models_h5/CNN_2_weights.h5")
