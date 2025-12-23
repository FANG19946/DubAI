# diarization.py
import os
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../../.env'))

hf_token = os.getenv("HUGGINGFACE_TOKEN")

base_rttm_path = "data/rttm/"  
base_wav_path = "data/wav/"  


# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=hf_token)

# run the pipeline on an audio file
diarization = pipeline( base_wav_path + "dialogue_diarization.wav")

# dump the diarization output to disk using RTTM format
with open(base_rttm_path + "diarization_10_to_15_v2.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
