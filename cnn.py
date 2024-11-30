import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate # type: ignore
from tensorflow.keras.losses import categorical_crossentropy, categorical_hinge, hinge, squared_hinge # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import tensorflow as tf

def load_data():


    x = np.load('dataset/Fer_X.npy')
    y = np.load('dataset/Fer_Y.npy')

    x = np.expand_dims(x, -1) # Add channel dimension to fit the input shape of the model
    x = x / 255.0 # Normalize
    y = np.eye(7,dtype='uint8')[y] # One hot encoding to fit the output shape of the model
    print(y.shape)
    print(y)

    Split = np.load('dataset/Fer_Usage.npy') # Load the split data
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')

    X_Train = x[x_index[0]:x_index[-1]+1]
    X_Valid = x[y_index[0]:y_index[-1]+1]
    X_Test  = x[z_index[0]:z_index[-1]+1]
    Y_Train = y[x_index[0]:x_index[-1]+1]
    Y_Valid = y[y_index[0]:y_index[-1]+1]
    Y_Test  = y[z_index[0]:z_index[-1]+1]

    # Save split data
    # np.save("Fer2013_X_train.npy",X_Train)
    # np.save("Fer2013_X_valid.npy",X_Valid)
    # np.save("Fer2013_X_test.npy",X_Test)
    # np.save("Fer2013_Y_train.npy",Y_Train)
    # np.save("Fer2013_Y_valid.npy",Y_Valid)
    # np.save("Fer2013_Y_test.npy",Y_Test)

    print(len(Y_Train))
    print(len(Y_Valid))
    print(len(Y_Test))
    return  X_Train,X_Test,X_Valid,Y_Train,Y_Test,Y_Valid


def CNN_v1():
    num_features = 64
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height = 48, 48
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data()

    model = Sequential()

    model.add(Conv2D(int(num_features/2), kernel_size=(5, 5), activation='relu', input_shape=(width, height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))


    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), # Changed 'lr' to 'learning_rate'
                  metrics=['accuracy'])

    filepath = "CNN_v1_Fer2013_best_weights.keras"
    early_stop = EarlyStopping(monitor='val_accuracy', patience=100,mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [early_stop,checkpoint]

    model_json = model.to_json()
    with open("CNN_v1_Fer2013_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit(
        data_generator.flow(X_Train, y_Train, batch_size=batch_size),
        steps_per_epoch=int(len(y_Train) / batch_size),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_Valid, y_Valid),
        shuffle=True
    )

    print("Model has been saved to disk ! Training time done !")


def CNN_v2():

    num_features = 64
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height = 48, 48
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data()

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(width, height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])



    filepath = "CNN_v2_2013_final_weights.keras"
    early_stop = EarlyStopping(monitor='val_loss', patience=20,mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stop, checkpoint]

    model_json = model.to_json()
    with open("CNN_v2_Fer2013_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit(data_generator.flow(X_Train,y_Train,
                                            batch_size=batch_size),
                                            steps_per_epoch=int(len(y_Train) / batch_size),
                                            epochs=epochs,
                                            verbose=1,
                                            callbacks=callbacks_list,
                                            validation_data=(X_Valid,y_Valid),
                                            shuffle=True
                        )

    # final_weight_path = "ConvNetV2_" + data_name[code] + "_final_weights.weights.h5"
    # model.save_weights(final_weight_path)

    print("Model has been saved to disk ! Training time done !")


def ExtractFeatures_Layer(dim):
    input_layer = Input(shape=(dim,))
    x = Dense(4096, kernel_regularizer=l2(0.01))(input_layer)
    x = Dropout(0.5)(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def CNN_Layer(width, height, depth):
    input_layer = Input(shape=(width, height, depth))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format='channels_last')(input_layer)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def create_dataset(X_img, X_sift, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((X_img, X_sift), y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE) # Changed 'tf.data.experimental.AUTOTUNE' to 'tf.data.AUTOTUNE'
    return dataset


def CNN_and_SIFT():
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height, depth = 48, 48, 1

    print("Loading Data !")
    X_Train, X_Test, X_Valid, y_Train, y_Test, y_Valid = load_data()


    Split = np.load('/content/drive/MyDrive/CV_data/Fer_Usage.npy')
    # Check the shape and data type of Split
    print("Shape of Split:", Split.shape)
    print("Data type of Split:", Split.dtype)
    print("Unique values in Split:", np.unique(Split))

    # Convert Split to string type if it's not already
    Split = Split.astype(str)

    x_index = np.where(Split == 'Training')[0]  # Get the indices directly as a 1D array
    y_index = np.where(Split == 'PublicTest')[0]  # Get the indices directly as a 1D array

    # Check if x_index and y_index are empty
    if x_index.size == 0:
        raise ValueError("No 'Training' entries found in Split array")
    if y_index.size == 0:
        raise ValueError("No 'PublicTest' entries found in Split array")

    X_SIFT = np.load("/Users/hiimbias/PycharmProjects/FED/models/Fer2013_SIFTDetector_Histogram_GEN.npy", allow_pickle=True)
    X_SIFT = X_SIFT.astype('float64')

    # Print the shape of X_SIFT before slicing
    print("Shape of X_SIFT before slicing:", X_SIFT.shape)

    if isinstance(X_SIFT, list):
        X_SIFT = np.vstack(X_SIFT)  # or np.concatenate(X_SIFT, axis=0) if they have the same shape
    # If X_SIFT has only one dimension, reshape it to have at least two dimensions
    elif X_SIFT.ndim == 1:
        X_SIFT = X_SIFT.reshape(-1, 1)  # Reshape to have one

    X_SIFT_Train = X_SIFT[x_index[0]:x_index[-1] + 1]
    X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1] + 1]


    print("Data has been generated !")

    SIFT = ExtractFeatures_Layer(X_SIFT_Train.shape[1])
    CNN = CNN_Layer(width, height, depth)

    MergeModel = concatenate([CNN.output, SIFT.output])

    m = Dense(2048, activation='relu')(MergeModel)
    m = Dropout(0.5)(m)
    m = Dense(num_labels, activation='softmax')(m)

    model = Model(inputs=[CNN.input, SIFT.input], outputs=m)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    filepath = "ConvSIFTNET_Fer2013_best_weights.keras"
    early_stop = EarlyStopping(monitor='val_acc', patience=50, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]

    model_json = model.to_json()
    with open("ConvSIFTNET_Fer2013.json", "w") as json_file:
        json_file.write(model_json)

    train_dataset = create_dataset(X_Train, X_SIFT_Train, y_Train, batch_size)
    valid_dataset = create_dataset(X_Valid, X_SIFT_Valid, y_Valid, batch_size)

    model.fit(train_dataset,
              steps_per_epoch=len(y_Train) // batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks_list,
              validation_data=valid_dataset,
              shuffle=True
              )

    final_weight_path = "ConvSIFTNET_Fer2013_final_weights.weights.h5"
    model.save_weights(final_weight_path)

    final_model_path = "ConvSIFTNET_Fer2013_final_model.keras"
    model.save(final_model_path)

    print("Model has been saved to disk ! Training time done !")



def main():
    CNN_v1()
    CNN_v2()
    CNN_and_SIFT()


if __name__ == "__main__":
    main()
