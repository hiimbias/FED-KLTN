import numpy as np

def load_data_CKplus():
    x = np.load('dataset/CK+/CK+_X.npy')
    y = np.load('dataset/CK+/CK+_Y.npy')

    x = np.expand_dims(x, -1)
    x = x / 255.0
    y = np.eye(7, dtype='uint8')[y]

    Split = np.load('dataset/CK+/CK+_Usage.npy')
    print("Unique values in Split:", np.unique(Split))  # Debug print

    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'Validation')
    z_index, = np.where(Split == 'Test')

    if len(y_index) == 0:
        raise ValueError("No 'Validation' entries found in Split array")

    X_Train = x[x_index[0]:x_index[-1] + 1]
    X_Valid = x[y_index[0]:y_index[-1] + 1]
    X_Test = x[z_index[0]:z_index[-1] + 1]
    Y_Train = y[x_index[0]:x_index[-1] + 1]
    Y_Valid = y[y_index[0]:y_index[-1] + 1]
    Y_Test = y[z_index[0]:z_index[-1] + 1]

    # np.save("CK+_X_train.npy", X_Train)
    # np.save("CK+_X_valid.npy", X_Valid)
    # np.save("CK+_X_test.npy", X_Test)
    # np.save("CK+_Y_train.npy", Y_Train)
    # np.save("CK+_Y_valid.npy", Y_Valid)
    # np.save("CK+_Y_test.npy", Y_Test)
    print(len(Y_Train))
    print(len(Y_Valid))
    print(len(Y_Test))

    return X_Train, X_Test, X_Valid, Y_Train, Y_Test, Y_Valid

load_data_CKplus()