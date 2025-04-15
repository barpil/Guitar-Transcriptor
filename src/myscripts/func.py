from pathlib import Path
import pandas as pd
import librosa

def load_dataset(data_path, sr=None):
    data = []
    feature_map = {
        's': 'steel',
        'n': 'nylon',
        'p': 'pick',
        'f': 'finger',
        'n': 'nail',
        'n': 'normal',
        'l': 'loud',
        'm': 'muted'
    }

    for file in Path(data_path).rglob("*.wav"):
        file_name = file.stem
        file_name_parts = file_name.split("-")
        clazz = file_name_parts[0]
        feature_code = file_name_parts[2]

        if len(feature_code) == 3:
            string = feature_map.get(feature_code[0], 'unknown')
            pluck = feature_map.get(feature_code[1], 'unknown')
            sound = feature_map.get(feature_code[2], 'unknown')
            try:
                y, sr = librosa.load(file, sr=sr)
            except Exception as e:
                print(f"Error: {file}:{e}")
                continue
            data.append([file_name, y, sr, string, pluck, sound, clazz])

    df_raw = pd.DataFrame(data, columns=["file_name", "audio_file", "sampling_rate", "string_type", "pluck_type",
                                         "sound_type", "clazz"])
    df = pd.get_dummies(df_raw, columns=["string_type", "pluck_type", "sound_type"])
    return df


def _audio_features_extraction(df_row, n_fft=2048, hop_length=512):
    audio = df_row["audio_file"]
    sr_v = df_row["sampling_rate"]

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr_v, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr_v, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr_v, n_fft=n_fft, hop_length=hop_length)

    tmp = df_row["file_name"].split("-")
    label = tmp[0] + '_' + tmp[2]
    return mel_spectrogram, chroma, contrast, label


def process_dataset(df, n_fft=2048, hop_lenght=512):
    mel_t=[]
    chroma_t=[]
    contrast_t=[]
    labels = []
    for _, df_row in df.iterrows():
        mel, chroma, contrast, label = _audio_features_extraction(df_row, n_fft, hop_lenght)
        mel_t.append(mel)
        chroma_t.append(chroma)
        contrast_t.append(contrast)
        labels.append(label)
    return mel_t, chroma_t, contrast_t, labels