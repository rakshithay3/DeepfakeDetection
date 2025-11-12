import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

def create_video_model(sequence_length=16, img_size=112):
    """
    Create 3D CNN for video analysis
    """
    inputs = keras.Input(shape=(sequence_length, img_size, img_size, 3))
    
    # 3D Convolutional layers
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # LSTM for temporal features
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

def extract_video_frames(video_path, num_frames=16, img_size=112):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        # Pad with last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    return np.array(frames)

def train_video_model(train_dir, epochs=50):
    """Train video model"""
    print("Training Video Deepfake Detection Model")
    
    model = create_video_model()
    
    # Load and prepare data (implement your data loading)
    # X_train, y_train = load_video_dataset(train_dir)
    
    # Train model
    # history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    
    model.save('models/video_model.h5')
    print("Video model saved!")
    
    return model

if __name__ == '__main__':
    train_video_model('data/videos')
