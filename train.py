import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from utils import load_data, split_data, create_model

def train_model(X, y, test_size=0.1, valid_size=0.1, batch_size=64, epochs=100):
    data = split_data(X, y, test_size=test_size, valid_size=valid_size)
    
    model = create_model()
    
    # use tensorboard to view metrics
    tensorboard = TensorBoard(log_dir="logs")
    early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)
    
    # training the model
    model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size,
              validation_data=(data["X_valid"], data["y_valid"]),
              callbacks=[tensorboard, early_stopping])
    
    model.save("results/model.h5")
    
    # evaluating the model
    print(f"Evaluating the model using {len(data['X_test'])} samples...")
    loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)
