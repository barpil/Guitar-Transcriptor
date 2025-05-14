from typing import BinaryIO, Dict, Any

import joblib
import librosa
import numpy as np
import pandas as pd
import torch

import config
from src.myscripts import model

#sr =22050

class PredictionModel():
    def __init__(self, sound_rec_pth: str, string_rec_pth: str, sound_type_rec_pth: str,
                 sr: int, n_fft: int = 2048, hop_length: int = 1024):  # Zmieniono domyślny hop_length na 1024
        """
        Inicjalizuje PredictionModel.

        Args:
            sound_rec_pth (str): Ścieżka do pliku modelu rozpoznawania dźwięku (.pth).
            string_rec_pth (str): Ścieżka do pliku modelu rozpoznawania struny (.pth).
            sound_type_rec_pth (str): Ścieżka do pliku modelu rozpoznawania typu dźwięku (.pth).
            sr (int): Częstotliwość próbkowania.
            n_fft (int): Długość okna FFT. Domyślnie 2048.
            hop_length (int): Długość przesunięcia okna. Domyślnie 1024 (zgodnie z treningiem).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Oczekiwane długości sekwencji na podstawie inicjalizacji modeli w skrypcie treningowym
        self.expected_seq_len_sound_and_string = 5632
        self.expected_seq_len_sound_type = 836

        self.sound_model = model.Conv1DClassifier(num_classes=43, input_shape=(self.expected_seq_len_sound_and_string, 1))
        self.sound_model.load_state_dict(
            torch.load(f"{config.MODEL_DIR_PATH}/{sound_rec_pth}", map_location=self.device))
        self.sound_model.to(self.device)
        self.sound_model.eval()

        self.string_model = model.Conv1DClassifier(num_classes=2, input_shape=(self.expected_seq_len_sound_and_string, 1))
        self.string_model.load_state_dict(
            torch.load(f"{config.MODEL_DIR_PATH}/{string_rec_pth}", map_location=self.device))
        self.string_model.to(self.device)
        self.string_model.eval()

        self.sound_type_model = model.Conv1DClassifier(num_classes=3, input_shape=(self.expected_seq_len_sound_type, 1))
        self.sound_type_model.load_state_dict(
            torch.load(f"{config.MODEL_DIR_PATH}/{sound_type_rec_pth}", map_location=self.device))
        self.sound_type_model.to(self.device)
        self.sound_type_model.eval()

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

        try:
            self.sound_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/sound_encoder.joblib")
            self.string_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/string_encoder.joblib")
            self.sound_type_encoder = joblib.load(f"{config.MODEL_DIR_PATH}/sound_type_encoder.joblib")
        except FileNotFoundError as e:
            print(f"Error: Encoder file not found: {e}")
            raise
        except Exception as e:
            print(f"Error while loading saved encoder: {e}")
            raise

    def _preprocess_features(self, features_flat: np.ndarray, expected_len: int) -> np.ndarray:
        """
        Dopasowuje długość spłaszczonych cech do oczekiwanej długości przez model
        poprzez dopełnianie zerami lub przycinanie.
        W przyszlosci powinienem dodac tutaj obsluge probek dluzszych w taki sposob ze robi predykcje
        dla listy kolejnych 2 sekundowych probek.
        """
        if len(features_flat) < expected_len:
            padding = np.zeros(expected_len - len(features_flat))
            features_flat = np.concatenate((features_flat, padding))
        elif len(features_flat) > expected_len:
            features_flat = features_flat[:expected_len]
        return features_flat

    def predict(self, sound_file_path: str) -> Dict[str, Any]:
        try:
            audio, sr = librosa.load(sound_file_path, sr=self.sr)
        except Exception as e:
            print(f"Error while loading audiofile {sound_file_path}: {e}")
            raise

        mel_spectrogram_flat = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        ).flatten()
        chroma_flat = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        ).flatten()
        contrast_flat = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        ).flatten()

        mel_spectrogram_processed = self._preprocess_features(mel_spectrogram_flat, self.expected_seq_len_sound_and_string)
        sound_type_features_flat = np.concatenate((chroma_flat, contrast_flat))
        sound_type_features_processed = self._preprocess_features(sound_type_features_flat,
                                                                  self.expected_seq_len_sound_type)

        sound_input_np = mel_spectrogram_processed.reshape(1, self.expected_seq_len_sound_and_string, 1)
        string_input_np = sound_input_np # Bo i tak te same dla sound i string
        sound_type_input_np = sound_type_features_processed.reshape(1, self.expected_seq_len_sound_type, 1)

        sound_tensor = torch.tensor(sound_input_np, dtype=torch.float32).to(self.device)
        string_tensor = torch.tensor(string_input_np, dtype=torch.float32).to(self.device)
        sound_type_tensor = torch.tensor(sound_type_input_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_sound_logits = self.sound_model(sound_tensor)
            pred_sound_idx = torch.argmax(pred_sound_logits, dim=1).cpu().numpy()
            pred_sound = self.sound_encoder.inverse_transform(pred_sound_idx)

            pred_string_logits = self.string_model(string_tensor)
            pred_string_idx = torch.argmax(pred_string_logits, dim=1).cpu().numpy()
            pred_string = self.string_encoder.inverse_transform(pred_string_idx)

            pred_sound_type_logits = self.sound_type_model(sound_type_tensor)
            pred_sound_type_idx = torch.argmax(pred_sound_type_logits, dim=1).cpu().numpy()
            pred_sound_type = self.sound_type_encoder.inverse_transform(pred_sound_type_idx)

        return {
            "sound": pred_sound[0] if len(pred_sound) > 0 else None,
            "string": pred_string[0] if len(pred_string) > 0 else None,
            "sound_type": pred_sound_type[0] if len(pred_sound_type) > 0 else None
        }