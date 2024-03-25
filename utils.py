import pandas as pd
import numpy as np
import os
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

label2int = {"male": 1, "female": 0}

def load_data(vector_length=128):
    if not os.path.isdir("results"):
        os.mkdir("results")

    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        return np.load("results/features.npy"), np.load("results/labels.npy")

    df = pd.read_csv("balanced-all.csv")
    n_samples = len(df)
    print("Total samples:", n_samples)
    print("Total male samples:", df['gender'].value_counts()['male'])
    print("Total female samples:", df['gender'].value_counts()['female'])

    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))

    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        X[i] = np.load(filename)
        y[i] = label2int[gender]

    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y

def split_data(X, y, test_size=0.1, valid_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

def create_model(vector_length=128):
    model = Sequential([
        Dense(256, input_shape=(vector_length,), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    model.summary()
    return model
