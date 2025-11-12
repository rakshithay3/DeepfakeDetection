import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import librosa

def create_audio_model(input_shape=(128, 128, 1)):
    """
    Create CNN model for audio analysis
    Uses mel-spectrogram as input
    """
    inputs = keras.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
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

def extract_audio_features_for_training(audio_path, n_mels=128, duration=3):
    """Extract mel-spectrogram from audio"""
    y, sr = librosa.load(audio_path, duration=duration)
    
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Resize to fixed size
    mel_spec_resized = cv2.resize(mel_spec_db, (128, 128))
    mel_spec_resized = np.expand_dims(mel_spec_resized, axis=-1)
    
    return mel_spec_resized

def train_audio_model(train_dir, epochs=50):
    """Train audio model"""
    print("Training Audio Deepfake Detection Model")
    
    model = create_audio_model()
    
    # Load and prepare data (implement your data loading)
    # X_train, y_train = load_audio_dataset(train_dir)
    
    # Train model
    # history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    
    model.save('models/audio_model.h5')
    print("Audio model saved!")
    
    return model

if __name__ == '__main__':
    train_audio_model('data/audio')
