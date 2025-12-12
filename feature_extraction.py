import os
import numpy as np
import pandas as pd
import librosa
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)


def extract_features(file_path,
                     sr=22050, # Sample rate
                     target_duration=5.0,
                     n_fft=2048,
                     n_mels=128,
                     n_mfcc=20):

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, mono=True)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Target length in samples
        target_length = int(sr * target_duration)

        # Pad or trim to target length
        if len(y) < target_length:
            # Pad with zeros
            y = librosa.util.fix_length(y, size=target_length)
        else:
            # Trim to target length (take middle portion)
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]

        # Adjust n_fft if needed
        effective_n_fft = min(n_fft, len(y))
        # Ensure n_fft is power of 2
        effective_n_fft = 2 ** int(np.log2(effective_n_fft))
        # For example, if effective_n_fft is 300,
        # it will become 256 after this step (the power of 2 that is closest to and not greater than 300).
        effective_hop_length = effective_n_fft // 4

        # === Time-domain features ===
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=effective_n_fft,
                                                 hop_length=effective_hop_length)[0]
        rms = librosa.feature.rms(y=y, frame_length=effective_n_fft,
                                  hop_length=effective_hop_length)[0]

        # === Spectral features ===
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=effective_n_fft,
                                                     hop_length=effective_hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=effective_n_fft,
                                                   hop_length=effective_hop_length,
                                                   roll_percent=0.85)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=effective_n_fft,
                                                       hop_length=effective_hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=effective_n_fft,
                                                     hop_length=effective_hop_length)[0]

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=effective_n_fft,
                                                     hop_length=effective_hop_length)

        # === Mel spectrogram ===
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=effective_n_fft,
                                                  hop_length=effective_hop_length,
                                                  n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # === MFCCs and deltas ===
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # === Chroma features ===
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=effective_n_fft,
                                             hop_length=effective_hop_length)

        # === Tonnetz (harmonic content) ===
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

        # === Aggregate statistics ===
        features = []

        # Time-domain: mean, std, max, min
        for feat in [zcr, rms]:
            features.extend([np.mean(feat), np.std(feat), np.max(feat), np.min(feat)])

        # Spectral features: mean, std, max, min
        for feat in [centroid, rolloff, bandwidth, flatness]:
            features.extend([np.mean(feat), np.std(feat), np.max(feat), np.min(feat)])

        # Spectral contrast: mean and std for each band
        for i in range(contrast.shape[0]):
            features.extend([np.mean(contrast[i]), np.std(contrast[i])])

        # MFCCs: mean and std
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # MFCC deltas: mean and std
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        # MFCC delta-deltas: mean only
        features.extend(np.mean(mfcc_delta2, axis=1))

        # Chroma: mean and std
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Tonnetz: mean and std
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))

        # Mel spectrogram global stats
        features.extend([np.mean(mel_db), np.std(mel_db),
                         np.max(mel_db), np.min(mel_db)])

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        # Return zero vector if extraction fails
        return np.zeros(216, dtype=np.float32)  # Adjust size based on feature count


def build_feature_dataset(root_dir, out_csv='features_robust.csv', target_duration=5.0):

    rows = []
    labels = []
    files = []
    failed_files = []

    print("=" * 70)
    print("ROBUST FEATURE EXTRACTION FOR ESC-22")
    print("=" * 70)
    print(f"Target duration: {target_duration} seconds")
    print(f"Sampling rate: 22050 Hz")
    print(f"MFCCs: 20 coefficients")

    # Get categories
    categories = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])

    # Count total files
    total_files = 0
    for label in categories:
        lab_dir = os.path.join(root_dir, label)
        audio_files = [f for f in os.listdir(lab_dir)
                       if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]
        total_files += len(audio_files)

    print(f"Found {len(categories)} categories")
    print(f"Total audio files: {total_files}")
    print("=" * 70 + "\n")

    # Process files
    processed = 0
    for label in categories:
        lab_dir = os.path.join(root_dir, label)
        audio_files = [f for f in os.listdir(lab_dir)
                       if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg'))]

        category_count = 0
        for fname in audio_files:
            fp = os.path.join(lab_dir, fname)

            feat = extract_features(fp, target_duration=target_duration)

            # Check if extraction was successful (non-zero features)
            if np.any(feat):
                rows.append(feat)
                labels.append(label)
                files.append(fp)
                category_count += 1
            else:
                failed_files.append(fp)

            # Print progress every time 100 audio files are processed
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_files} files...")

        print(f"  ✓ {label}: {category_count} samples extracted")

    print("\n" + "=" * 70)
    print(f"Successfully processed: {len(rows)}/{total_files} files")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for fp in failed_files[:5]:
            print(f"  - {fp}")
    print("=" * 70)

    # Convert to DataFrame
    feat_arr = np.vstack(rows)

    # Generate column names (216 total features)
    col_names = []

    # Time-domain (2 × 4)
    for feat in ['zcr', 'rms']:
        for stat in ['mean', 'std', 'max', 'min']:
            col_names.append(f"{feat}_{stat}")

    # Spectral (4 × 4)
    for feat in ['centroid', 'rolloff', 'bandwidth', 'flatness']:
        for stat in ['mean', 'std', 'max', 'min']:
            col_names.append(f"{feat}_{stat}")

    # Spectral contrast (7 × 2)
    for i in range(7):
        for stat in ['mean', 'std']:
            col_names.append(f"contrast_band{i + 1}_{stat}")

    # MFCCs (20 × 2)
    for i in range(20):
        col_names.append(f"mfcc{i + 1}_mean")
    for i in range(20):
        col_names.append(f"mfcc{i + 1}_std")

    # MFCC deltas (20 × 2)
    for i in range(20):
        col_names.append(f"mfcc_delta{i + 1}_mean")
    for i in range(20):
        col_names.append(f"mfcc_delta{i + 1}_std")

    # MFCC delta-deltas (20 × 1)
    for i in range(20):
        col_names.append(f"mfcc_delta2_{i + 1}_mean")

    # Chroma (12 × 2)
    for i in range(12):
        for stat in ['mean', 'std']:
            col_names.append(f"chroma{i + 1}_{stat}")

    # Tonnetz (6 × 2)
    for i in range(6):
        for stat in ['mean', 'std']:
            col_names.append(f"tonnetz{i + 1}_{stat}")

    # Mel stats (4)
    for stat in ['mean', 'std', 'max', 'min']:
        col_names.append(f"mel_db_{stat}")

    # Create DataFrame
    df = pd.DataFrame(feat_arr, columns=col_names[:feat_arr.shape[1]])
    df['label'] = labels
    df['file'] = files

    # Save
    df.to_csv(out_csv, index=False)

    print(f"\n✓ Feature dataset saved to: {out_csv}")
    print("=" * 70)

    # Summary
    print("\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features per sample: {feat_arr.shape[1]}")
    print(f"  Number of classes: {df['label'].nunique()}")
    print(f"\nClass distribution:")
    class_counts = df['label'].value_counts().sort_index()
    print(class_counts)

    # Check for imbalanced classes
    min_count = class_counts.min()
    max_count = class_counts.max()
    if max_count > min_count * 1.5:
        print(f"\n⚠ Warning: Class imbalance detected!")
        print(f"  Min samples per class: {min_count}")
        print(f"  Max samples per class: {max_count}")
        print(f"  Ratio: {max_count / min_count:.2f}x")

    return df


if __name__ == "__main__":
    # ESC-22 dataset path
    root_dir = './audios in the park'

    # Extract features
    df = build_feature_dataset(root_dir, out_csv="features_robust.csv", target_duration=5.0)

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)

    # Show sample data
    print("\nFirst 5 rows (first 10 features):")
    print(df.iloc[:5, :10])

    print("\nFeature statistics (first 10 features):")
    print(df.iloc[:, :10].describe())
