import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import L2


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# HYPERPARAMETERS
BATCH_SIZE = 32
PATH_TO_TRAIN_DATA = "Data/Processed/train/"
PATH_TO_TEST_DATA = "Data/Processed/test/"


def scaler(x_array):
    x_array = (x_array / 255 * 0.8) + 0.1
    return x_array


# PREPARE TRAIN AND TEST SETS
face_types = {0: "Heart", 1: "Oblong", 2: "Oval", 3: "Round", 4: "Square"}
X_train, X_test, y_train, y_test = [], [], [], []

for y, face_type in face_types.items():
    for file in os.listdir(PATH_TO_TRAIN_DATA + f'/{face_type}'):
        img = cv2.imread(PATH_TO_TRAIN_DATA + f'/{face_type}' + f'/{file}', cv2.IMREAD_GRAYSCALE)
        X_train.append(scaler(img))
        one_hot_encoding_vector = [0 for _ in range(len(face_types))]
        one_hot_encoding_vector[y] = 1
        y_train.append(one_hot_encoding_vector)

for y, face_type in face_types.items():
    for file in os.listdir(PATH_TO_TEST_DATA + f'/{face_type}'):
        img = cv2.imread(PATH_TO_TEST_DATA + f'/{face_type}' + f'/{file}', cv2.IMREAD_GRAYSCALE)
        X_test.append(scaler(img))
        one_hot_encoding_vector = [0 for _ in range(len(face_types))]
        one_hot_encoding_vector[y] = 1
        y_test.append(one_hot_encoding_vector)

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
print("Data preprocessing completed...")


# BUILD CNN MODEL
def build_model():
    """model from Kaggle https://www.kaggle.com/bradwel/facial-classification-model"""
    input_img = Input(shape=(224, 168, 1), name='ImageInput')
    x = Conv2D(16, (5, 5), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv1_1')(input_img)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(16, (3, 3), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = SeparableConv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv2_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    # x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv4_1')(x)
    # x = BatchNormalization(name='bn4')(x)
    # x = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2(0.00001), padding='same', name='Conv4_2')(x)
    # x = MaxPooling2D((2, 2), name='pool3')(x)
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
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dropout(0.2, name='dropout1')(x)
    x = Dense(32, activation='relu', name='fc2')(x)
    x = Dropout(0.2, name='dropout2')(x)
    x = Dense(5, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model


# Define model
ConvNet_model = build_model()
print(ConvNet_model.summary())

# Compile model
adam_optimizer = Adam(learning_rate=0.0001, decay=1e-5)
es = EarlyStopping(patience=5, min_delta=0.01, monitor="val_accuracy")
# chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
ConvNet_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_optimizer)

# Fit model
history = ConvNet_model.fit(x=X_train,
                            y=y_train,
                            validation_data=(X_test, y_test),
                            epochs=20,
                            callbacks=[es]
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
# ConvNet_model.save_weights("TrainingModels/models_h5/CNN_2_weights.h5")
