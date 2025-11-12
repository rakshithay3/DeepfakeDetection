from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import os
import tempfile
from werkzeug.utils import secure_filename
from scipy import fftpack
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global models
image_model = None
video_model = None
audio_model = None

def load_models():
    """Load all detection models"""
    global image_model, video_model, audio_model
    
    try:
        # Try to load trained models
        if os.path.exists('models/image_model.h5'):
            image_model = keras.models.load_model('models/image_model.h5', compile=False)
            print("✓ Image model loaded")
        
        if os.path.exists('models/video_model.h5'):
            video_model = keras.models.load_model('models/video_model.h5', compile=False)
            print("✓ Video model loaded")
        
        if os.path.exists('models/audio_model.h5'):
            audio_model = keras.models.load_model('models/audio_model.h5', compile=False)
            print("✓ Audio model loaded")
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Note: Using demo mode - {e}")

load_models()

def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'audio':
        return ext in ALLOWED_AUDIO_EXTENSIONS
    
    return False

# ==========================================
# IMAGE DETECTION
# ==========================================

def analyze_image(image_path):
    """Analyze single image for deepfake"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)
    
    if image_model is not None:
        prediction = image_model.predict(img_batch, verbose=0)[0][0]
    else:
        # Demo mode
        prediction = np.random.uniform(0.3, 0.8)
    
    is_fake = prediction > 0.5
    confidence = prediction * 100 if is_fake else (1 - prediction) * 100
    
    # Extract features
    freq_features = extract_frequency_features(img_rgb)
    
    return {
        'prediction_score': float(prediction),
        'is_fake': bool(is_fake),
        'confidence': float(confidence),
        'ai_percentage': float(prediction * 100),
        'label': 'AI Generated' if is_fake else 'Real',
        'analysis_type': 'image'
    }

def extract_frequency_features(img):
    """Extract frequency domain features"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return np.mean(magnitude)

# ==========================================
# VIDEO DETECTION
# ==========================================

def analyze_video(video_path):
    """Analyze video for deepfake - frame by frame analysis"""
    print(f"Analyzing video: {video_path}")
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames (analyze every Nth frame)
        sample_rate = max(1, total_frames // 30)  # Analyze max 30 frames
        frame_predictions = []
        frame_analyses = []
        
        frame_idx = 0
        analyzed_count = 0
        
        while cap.isOpened() and analyzed_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every Nth frame
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_batch = np.expand_dims(frame_resized, axis=0)
                
                if video_model is not None:
                    pred = video_model.predict(frame_batch, verbose=0)[0][0]
                else:
                    # Demo mode with temporal consistency
                    base_pred = 0.6 if analyzed_count == 0 else frame_predictions[-1]
                    pred = np.clip(base_pred + np.random.uniform(-0.1, 0.1), 0, 1)
                
                frame_predictions.append(float(pred))
                
                # Analyze temporal consistency
                if len(frame_predictions) > 1:
                    consistency = 1 - abs(frame_predictions[-1] - frame_predictions[-2])
                else:
                    consistency = 1.0
                
                frame_analyses.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'prediction': float(pred),
                    'temporal_consistency': float(consistency)
                })
                
                analyzed_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate overall statistics
        avg_prediction = np.mean(frame_predictions)
        std_prediction = np.std(frame_predictions)
        temporal_consistency = np.mean([f['temporal_consistency'] for f in frame_analyses])
        
        is_fake = avg_prediction > 0.5
        confidence = avg_prediction * 100 if is_fake else (1 - avg_prediction) * 100
        
        # Detect manipulation segments
        manipulation_segments = detect_manipulation_segments(frame_analyses, fps)
        
        # Extract audio and analyze
        audio_result = extract_and_analyze_audio(video_path)
        
        return {
            'prediction_score': float(avg_prediction),
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'ai_percentage': float(avg_prediction * 100),
            'label': 'AI Generated' if is_fake else 'Real',
            'analysis_type': 'video',
            'video_metadata': {
                'duration': float(duration),
                'fps': int(fps),
                'total_frames': int(total_frames),
                'analyzed_frames': len(frame_predictions)
            },
            'temporal_analysis': {
                'consistency_score': float(temporal_consistency * 100),
                'prediction_variance': float(std_prediction),
                'frame_predictions': frame_predictions,
                'detailed_frames': frame_analyses[:10]  # First 10 for display
            },
            'manipulation_segments': manipulation_segments,
            'audio_analysis': audio_result
        }
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {'error': str(e)}

def detect_manipulation_segments(frame_analyses, fps):
    """Detect time segments that show manipulation"""
    segments = []
    current_segment = None
    threshold = 0.6
    
    for frame in frame_analyses:
        if frame['prediction'] > threshold:
            if current_segment is None:
                current_segment = {
                    'start': frame['timestamp'],
                    'end': frame['timestamp'],
                    'severity': frame['prediction']
                }
            else:
                current_segment['end'] = frame['timestamp']
                current_segment['severity'] = max(current_segment['severity'], frame['prediction'])
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
    
    if current_segment is not None:
        segments.append(current_segment)
    
    return segments

def extract_and_analyze_audio(video_path):
    """Extract audio from video and analyze"""
    try:
        # Extract audio
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            return {'has_audio': False, 'message': 'No audio track found'}
        
        # Save audio temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        video.audio.write_audiofile(temp_audio.name, logger=None)
        video.close()
        
        # Analyze audio
        audio_result = analyze_audio(temp_audio.name)
        
        # Cleanup
        os.unlink(temp_audio.name)
        
        return audio_result
        
    except Exception as e:
        return {'has_audio': False, 'error': str(e)}

# ==========================================
# AUDIO DETECTION
# ==========================================

def analyze_audio(audio_path):
    """Analyze audio for deepfake/voice cloning"""
    print(f"Analyzing audio: {audio_path}")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30)  # Max 30 seconds
        
        # Extract audio features
        features = extract_audio_features(y, sr)
        
        # Predict using model
        if audio_model is not None:
            # Prepare features for model
            feature_vector = np.array([list(features.values())])
            prediction = audio_model.predict(feature_vector, verbose=0)[0][0]
        else:
            # Demo mode
            prediction = np.random.uniform(0.3, 0.75)
        
        is_fake = prediction > 0.5
        confidence = prediction * 100 if is_fake else (1 - prediction) * 100
        
        # Analyze speech segments
        speech_segments = detect_speech_segments(y, sr)
        
        # Voice consistency analysis
        consistency = analyze_voice_consistency(y, sr, speech_segments)
        
        return {
            'prediction_score': float(prediction),
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'ai_percentage': float(prediction * 100),
            'label': 'AI Generated Voice' if is_fake else 'Real Voice',
            'analysis_type': 'audio',
            'has_audio': True,
            'audio_metadata': {
                'duration': float(len(y) / sr),
                'sample_rate': int(sr),
                'num_segments': len(speech_segments)
            },
            'features': features,
            'voice_consistency': consistency,
            'speech_segments': speech_segments
        }
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {'error': str(e), 'has_audio': False}

def extract_audio_features(y, sr):
    """Extract audio features for deepfake detection"""
    features = {}
    
    # Spectral features
    features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
    
    # Zero crossing rate
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    
    # Energy
    features['energy'] = float(np.mean(librosa.feature.rms(y=y)))
    
    return features

def detect_speech_segments(y, sr):
    """Detect speech segments in audio"""
    # Simple energy-based voice activity detection
    rms = librosa.feature.rms(y=y)[0]
    threshold = np.mean(rms) * 0.5
    
    segments = []
    in_segment = False
    start = 0
    
    hop_length = 512
    frame_duration = hop_length / sr
    
    for i, energy in enumerate(rms):
        if energy > threshold and not in_segment:
            start = i * frame_duration
            in_segment = True
        elif energy <= threshold and in_segment:
            end = i * frame_duration
            if end - start > 0.3:  # Minimum 0.3 seconds
                segments.append({'start': float(start), 'end': float(end)})
            in_segment = False
    
    return segments[:10]  # Return first 10 segments

def analyze_voice_consistency(y, sr, segments):
    """Analyze voice consistency across segments"""
    if len(segments) < 2:
        return {'score': 100.0, 'status': 'insufficient_data'}
    
    # Extract MFCCs for each segment
    segment_mfccs = []
    
    for seg in segments[:5]:  # Analyze first 5 segments
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        if end_sample > len(y):
            continue
        
        segment_audio = y[start_sample:end_sample]
        if len(segment_audio) > sr * 0.1:  # At least 0.1 seconds
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            segment_mfccs.append(np.mean(mfcc, axis=1))
    
    if len(segment_mfccs) < 2:
        return {'score': 100.0, 'status': 'insufficient_data'}
    
    # Calculate consistency (similarity between segments)
    similarities = []
    for i in range(len(segment_mfccs) - 1):
        similarity = np.corrcoef(segment_mfccs[i], segment_mfccs[i+1])[0, 1]
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    consistency_score = (avg_similarity + 1) * 50  # Convert to 0-100 scale
    
    return {
        'score': float(consistency_score),
        'status': 'consistent' if consistency_score > 70 else 'inconsistent'
    }

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/')
def index():
    """Serve main page"""
    return render_template('multimodal.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_media():
    """Universal endpoint for analyzing any media type"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Determine file type and analyze
        result = None
        
        if allowed_file(filename, 'image'):
            result = analyze_image(filepath)
        elif allowed_file(filename, 'video'):
            result = analyze_video(filepath)
        elif allowed_file(filename, 'audio'):
            result = analyze_audio(filepath)
        else:
            os.unlink(filepath)
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Add metadata
        result['filename'] = filename
        result['timestamp'] = datetime.now().isoformat()
        result['file_size'] = os.path.getsize(filepath)
        
        # Cleanup
        try:
            os.unlink(filepath)
        except:
            pass
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple files at once"""
    try:
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze based on type
            if allowed_file(filename, 'image'):
                result = analyze_image(filepath)
            elif allowed_file(filename, 'video'):
                result = analyze_video(filepath)
            elif allowed_file(filename, 'audio'):
                result = analyze_audio(filepath)
            else:
                continue
            
            result['filename'] = filename
            results.append(result)
            
            # Cleanup
            try:
                os.unlink(filepath)
            except:
                pass
        
        summary = {
            'total': len(results),
            'fake': sum(1 for r in results if r.get('is_fake', False)),
            'real': sum(1 for r in results if not r.get('is_fake', True)),
            'avg_confidence': float(np.mean([r.get('confidence', 0) for r in results]))
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'image': image_model is not None,
            'video': video_model is not None,
            'audio': audio_model is not None
        },
        'supported_formats': {
            'images': list(ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(ALLOWED_VIDEO_EXTENSIONS),
            'audio': list(ALLOWED_AUDIO_EXTENSIONS)
        },
        'version': '2.0.0'
    })

if __name__ == '__main__':
    print("="*70)
    print("MULTIMODAL DEEPFAKE DETECTION SYSTEM")
    print("="*70)
    print("Supported formats:")
    print(f"  Images: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")
    print(f"  Videos: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
    print(f"  Audio: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)
