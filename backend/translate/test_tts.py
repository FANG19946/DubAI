import torch
from TTS.api import TTS


device = "cuda" if torch.cuda.is_available() else "cpu"

def make_audio(text = "This is my first TTS test"):
    tts = TTS(model_name = "tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)
    tts.tts_to_file(
                text="यह एक परीक्षण वाक्य है।",
                speaker_wav="clean_voice.wav",
                language="hi", 
                file_path = "output.wav")
    return "output.wav"

print(make_audio())