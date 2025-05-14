import config
from src.api.pipeline.model.prediction_model import PredictionModel

sound_rec_pth = f"sound_recognition_model.pth"
string_rec_pth = f"string_recognition_model.pth"
sound_type_rec_pth = f"sound_type_recognition_model.pth"
model = PredictionModel(sound_rec_pth, string_rec_pth, sound_type_rec_pth, 22050)
print("Starting prediction")
result = model.predict(f"{config.SOUNDS_DATA_DIR_PATH}/A2/A2-1-spn.wav")
print(result)