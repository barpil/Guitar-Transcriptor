from typing import BinaryIO

import joblib
import librosa
import pandas as pd
import torch

import config
from src.myscripts import model

#sr =22050

class PredictionModel():
    def __init__(self, sound_rec_pth, string_rec_pth, sound_type_rec_pth, sr, n_fft=2048, hop_length=512):
        self.sound_model = model.Conv1DClassifier(num_classes=43, input_shape=(5632, 1)) #43
        self.sound_model.load_state_dict(torch.load(f"{config.MODEL_DIR_PATH}/{sound_rec_pth}"))
        self.sound_model.eval()
        self.string_model = model.Conv1DClassifier(num_classes=2, input_shape=(5632, 1)) #2
        self.string_model.load_state_dict(torch.load(f"{config.MODEL_DIR_PATH}/{string_rec_pth}"))
        self.string_model.eval()
        self.sound_type_model = model.Conv1DClassifier(num_classes=3, input_shape=(836, 1)) #3
        self.sound_type_model.load_state_dict(torch.load(f"{config.MODEL_DIR_PATH}/{sound_type_rec_pth}"))
        self.sound_type_model.eval()

        self.sr = sr
        self.n_fft=n_fft
        self.hop_length=hop_length

    def predict(self, sound_file: BinaryIO):
        filename = getattr(sound_file, "name", None)
        if not filename or not filename.lower().endswith(".wav"):
            raise ValueError("Passed sound file must be in .wav format!")
        audio, sr = librosa.load(sound_file, sr=self.sr)

        mel_spectrogram = pd.DataFrame(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length))
        chroma = pd.DataFrame(librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length))
        contrast = pd.DataFrame(librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length))

        sound_tensor = torch.tensor(mel_spectrogram, dtype= torch.float32)
        string_tensor = torch.tensor(mel_spectrogram, dtype= torch.float32)
        sound_type_tensor = torch.tensor(pd.concat([chroma, contrast], axis=1), dtype= torch.float32)

        sound_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/sound_encoder.joblib")
        string_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/string_encoder.joblib")
        sound_type_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/sound_type_encoder.joblib")
        with torch.no_grad():
            pred_sound = sound_encoder.inverse_transform(torch.argmax(self.sound_model(sound_tensor), dim=1).cpu().numpy())
            pred_string = string_encoder.inverse_transform(torch.argmax(self.string_model(string_tensor), dim=1).cpu().numpy())
            pred_sound_type = sound_type_encoder.inverse_transform(torch.argmax(self.sound_type_model(sound_type_tensor), dim=1).cpu().numpy())

        return {"sound": pred_sound, "string": pred_string, "sound_type": pred_sound_type}