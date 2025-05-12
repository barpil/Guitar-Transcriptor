from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler


import config


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


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_feature_combination_dataframe(
    features_list: list,
    label_list: list,
    n_fft: int,
    hop_lenght: int,
    sr: int,
    data_path: str,
    save_dataframe: bool = False,
    filename: str = "features_dataframe",
):
    df = load_dataset(data_path, sr)
    scaler = StandardScaler()
    mel, chroma, contrast, labels = process_dataset(df, n_fft, hop_lenght)

    features_dict = {
        "mel": pd.DataFrame([row.flatten() for row in mel]),
        "chroma": pd.DataFrame([row.flatten() for row in chroma]),
        "contrast": pd.DataFrame([row.flatten() for row in contrast])
    }

    if not all(feature in features_dict for feature in features_list):
        raise ValueError(f"Feature in feature_list was not recognized.\nAvailable features:\n{features_dict.keys()}")

    # Rozdzielenie etykiet na podstawowe i złożone
    split_labels = map(lambda x: x.split("_", 1), labels)
    sound_labels, combined_3_labels = zip(*split_labels)
    sound_labels = list(sound_labels)
    combined_3_labels = list(combined_3_labels)

    string_labels, pluck_labels, sound_type_labels = zip(
        *map(lambda label: (label[0], label[1], label[2]), combined_3_labels)
    )
    string_labels = list(string_labels)
    pluck_labels = list(pluck_labels)
    sound_type_labels = list(sound_type_labels)

    labels_dict = {
        "sound": sound_labels,
        "string": string_labels,
        "pluck": pluck_labels,
        "sound_type": sound_type_labels
    }

    if not all(label in labels_dict for label in label_list):
        raise ValueError(f"Label in label_list was not recognized.\nAvailable labels:\n{labels_dict.keys()}")

    # Inicjalizacja etykiet
    if "sound" in label_list:
        result_labels = labels_dict["sound"]
    else:
        result_labels = [""] * len(labels)  # Pusta etykieta o poprawnej długości

    # Sklejanie pozostałych etykiet
    for label_key in ["string", "pluck", "sound_type"]:
        if label_key in label_list:
            result_labels = [x + "-" + y if x else y for x, y in zip(result_labels, labels_dict[label_key])]
    # Tworzenie końcowego DataFrame z wybranych cech
    result_dataframe = pd.concat(
        [features_dict[feature] for feature in features_list],
        axis=1
    )
    result_dataframe["label"] = result_labels

    if save_dataframe:
        result_dataframe.to_csv(f"{config.DATAFRAMES_DIR_PATH}/{filename}.csv", index=False)
    else:
        return result_dataframe
