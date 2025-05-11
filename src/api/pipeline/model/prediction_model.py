from typing import BinaryIO

import librosa
import torch

import config
from src.myscripts import model

#sr =22050

class PredictionModel():
    def __init__(self, sound_rec_pth, string_rec_pth, sound_type_rec_pth, sr):
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

    def predict(self, sound_file: BinaryIO):
        filename = getattr(sound_file, "name", None)
        if not filename or not filename.lower().endswith(".wav"):
            raise ValueError("Passed sound file must be in .wav format!")
        y, sr = librosa.load(sound_file, sr=self.sr)