from pydub import AudioSegment
from parse_srt import parse_srt
from dataclasses import dataclass
import torch
import torchaudio
import numpy as np
import io
from speechbrain.inference import SpeakerRecognition

import os

os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # <-- must be before anything else
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"



@dataclass
class DialogueSegment:
    index: int
    start: int  # in ms
    end: int    # in ms
    audio: AudioSegment

def gen_dialogue_array(audio_path, silence_duration=1000, srt_path = 'eng_10_to_15.srt' ,save_path = 'dialogue_diarization.wav' ):
    
    base_wav_path = "data/wav/"  
    base_srt_path = "data/srt/"  

    audio = AudioSegment.from_wav(base_wav_path + audio_path)
    subtitles = parse_srt(base_srt_path + srt_path)
    
    segments = []
    


    for i,sub in enumerate(subtitles):
        start = sub["start"]
        end = sub["end"]
        segment = audio[start:end]
        segments.append(DialogueSegment(i, start, end, segment))
    
    return segments
        


# Load the model once (move outside loop if calling multiple times)
embedding_model = SpeakerRecognition.from_hparams(
source="speechbrain/spkrec-ecapa-voxceleb",
savedir="pretrained_models/spkrec-ecapa-voxceleb",
run_opts={"local_file_strategy": "copy"},
)

def extract_embedding(audio_segment: AudioSegment):
    # Convert PyDub AudioSegment to raw bytes
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

    # Normalize (optional but helps if input wasn't already normalized)
    samples /= np.iinfo(audio_segment.array_type).max

    # Convert to mono if stereo
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
        samples = samples.mean(axis=1)

    # Convert to torch tensor with shape [1, time]
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)

   


    # Get embedding
    embedding = embedding_model.encode_batch(waveform)

    # Shape [1, 1, 192] -> squeeze to [192]
    return embedding.squeeze().detach().cpu().numpy()
