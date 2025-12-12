"""
Standalone script to predict environmental sound categories using a pre-trained model.

Usage:
    python userTest_predict.py /path/to/audio.wav
    python predict_audio.py /path/to/audio.wav --top_k 5
"""

import numpy as np
import librosa
import joblib
import argparse
import os
import warnings


def extract_features_from_audio(audio_path, sr=22050, n_fft=2048, hop_length=512,
                                n_mels=128, n_mfcc=20, min_duration=0.5):
    """
    Extract features matching the training pipeline.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Pad or trim to ensure minimum duration
    min_samples = int(min_duration * sr)
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # If after trimming it's too short, pad again
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')

    signal_length = len(y)

    # Dynamically adjust n_fft if needed
    if signal_length < n_fft:
        n_fft = 2 ** int(np.log2(signal_length))
        n_fft = max(512, n_fft)

    hop_length = n_fft // 4
    hop_length = min(hop_length, signal_length // 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # === Time-domain features ===
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

        # === Spectral features ===
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # === Mel spectrogram ===
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # === MFCCs and deltas ===
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # === Chroma features ===
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # === Tonnetz ===
        try:
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        except:
            tonnetz = np.zeros((6, max(1, len(y) // hop_length)))

    # === Aggregate statistics ===
    features = []

    def safe_stats(arr):
        if len(arr) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [np.mean(arr), np.std(arr), np.max(arr), np.min(arr)]

    # Time-domain: mean, std, max, min
    for feat in [zcr, rms]:
        features.extend(safe_stats(feat))

    # Spectral features: mean, std, max, min
    for feat in [centroid, rolloff, bandwidth, flatness]:
        features.extend(safe_stats(feat))

    # Spectral contrast: mean and std for each band
    for i in range(contrast.shape[0]):
        features.extend([np.mean(contrast[i]), np.std(contrast[i])])

    # MFCCs: mean and std
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # MFCC deltas: mean and std
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    # MFCC delta-deltas: only mean
    features.extend(np.mean(mfcc_delta2, axis=1))

    # Chroma: mean and std
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Tonnetz: mean and std
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz, axis=1))

    # Mel spectrogram global stats
    features.extend([np.mean(mel_db), np.std(mel_db), np.max(mel_db), np.min(mel_db)])

    return np.array(features, dtype=np.float32).reshape(1, -1)


def load_model_and_preprocessors(model_dir='.'):
    """
    Load saved model and preprocessing objects.

    Args:
        model_dir: Directory containing the saved .pkl files

    Returns:
        model, scaler, label_encoder, pca (or None if not used)
    """
    model_path = os.path.join(model_dir, 'best_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    pca_path = os.path.join(model_dir, 'pca.pkl')

    # Check required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found: {le_path}")

    print("Loading model and preprocessors...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)

    # PCA is optional
    pca = None
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        print("  ✓ Loaded: model, scaler, label_encoder, pca")
    else:
        print("  ✓ Loaded: model, scaler, label_encoder (no PCA)")

    return model, scaler, label_encoder, pca


def predict_audio(audio_path, model, scaler, label_encoder, pca=None, top_k=5):
    """
    Predict the environmental sound category for an audio file.

    Args:
        audio_path: Path to audio file
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: LabelEncoder for class names
        pca: Fitted PCA object (or None)
        top_k: Number of top predictions to show

    Returns:
        predicted_class, top_predictions (list of tuples)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get audio duration for display
    duration = librosa.get_duration(path=audio_path)

    print(f"\nAnalyzing: {os.path.basename(audio_path)}")
    print(f"Duration: {duration:.2f} seconds")

    # Extract features
    print("Extracting features...")
    features = extract_features_from_audio(audio_path)
    print(f"  ✓ Extracted {features.shape[1]} features")

    # Scale
    features_scaled = scaler.transform(features)

    # Apply PCA if used
    if pca is not None:
        features_scaled = pca.transform(features_scaled)
        print(f"  ✓ Applied PCA: {features_scaled.shape[1]} components")

    # Predict
    prediction = model.predict(features_scaled)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]

    # Get top-k predictions with probabilities
    top_predictions = []
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        for idx in top_indices:
            class_name = label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx] * 100
            top_predictions.append((class_name, confidence))
    else:
        # If model doesn't support probabilities, just return the prediction
        top_predictions.append((predicted_class, 100.0))

    return predicted_class, top_predictions


def main():
    parser = argparse.ArgumentParser(
        description='Predict environmental sound category for an audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict_audio.py dog_bark.wav
    python predict_audio.py rain.wav --top_k 3
    python predict_audio.py sounds/traffic.wav --model_dir ./trained_models/
        """
    )
    parser.add_argument('audio', type=str,
                       help='Path to audio file (WAV, MP3, FLAC, etc.)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to display (default: 5)')
    parser.add_argument('--model_dir', type=str, default='.',
                       help='Directory containing saved model files (default: current directory)')

    args = parser.parse_args()

    print("="*70)
    print("ENVIRONMENTAL SOUND CLASSIFIER")
    print("="*70)

    try:
        # Load model and preprocessors
        model, scaler, label_encoder, pca = load_model_and_preprocessors(args.model_dir)

        # Predict
        predicted_class, top_predictions = predict_audio(
            args.audio, model, scaler, label_encoder, pca, args.top_k
        )

        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\n🎵 Top Prediction: {predicted_class.upper()}")

        if len(top_predictions) > 1:
            print(f"\n📊 Top {args.top_k} Predictions:")
            print("-" * 50)
            for i, (class_name, confidence) in enumerate(top_predictions, 1):
                bar_length = int(confidence / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"{i}. {class_name:25s} {bar} {confidence:5.2f}%")

        print("="*70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've trained the model first by running:")
        print("  python Multiple_ClassificationModel_training.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()