import torch
from TTS.api import TTS
from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
import io
import re
# Testing
import json




device = "cuda" if torch.cuda.is_available() else "cpu"
base_json_path = "data/json/"  



def generate_audio(subtitles, speaker_wav, input_audio_path, output_audio_path):
    """
    Replaces segments in input audio using XTTS-dubbed subtitles.
    """
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(
        device
    )

    original = AudioSegment.from_wav(input_audio_path)
    frame_rate = original.frame_rate
    audio_duration = len(original)

 

    offset = 0

    for i, sub in enumerate(subtitles):
        text = sub["text"]


        # Original times
        original_start = sub["start"]
        original_end = sub["end"]
        duration = original_end - original_start

        
        # Shifted positions in the edited audio
        start = original_start + offset
        end = original_end + offset


        

        end = min(end, audio_duration)
        if start >= audio_duration + offset:
            break

        if duration <= 0 or not is_speakable(text):
            print(f"⚠️ Skipping non-speakable subtitle at index {sub.get('index', i)}: {repr(text)}")
            continue

        # Generate audio with XTTS
        wav_array = tts.tts(text=text, speaker_wav=speaker_wav, language="hi")

        # 🛠 Handle list or scalar output safely
        if isinstance(wav_array, list):
            if isinstance(wav_array[0], (np.float32, float)):
                wav_array = np.array(wav_array, dtype=np.float32)
            else:
                wav_array = wav_array[0]
        elif not isinstance(wav_array, np.ndarray):
            print(f"❌ Unexpected XTTS output format at index {i}. Skipping.")
            continue

        # 🔒 Guard against invalid scalar returns
        if wav_array.ndim == 0 or wav_array.shape == ():
            print(f"⚠️ Empty or scalar XTTS output at index {i}. Skipping.")
            continue

        # 🔁 Convert float32 [-1.0, 1.0] → int16 PCM
        wav_array = (wav_array * 32767).astype("int16")


        dubbed = AudioSegment(
            wav_array.tobytes(),
            frame_rate=tts.synthesizer.output_sample_rate,
            sample_width=2,
            channels=1,
        ).set_frame_rate(frame_rate)


        # Replace segment in original
        original = original[:start] + dubbed + original[end:]

        # Compute offset introduced by mismatch
        actual_duration = len(dubbed)
        offset += actual_duration - duration

        sub["dub_start"] = start
        sub["dub_end"] = start + actual_duration

    original.export(output_audio_path, format="wav")
    print(f"✅ Dubbed audio exported to: {output_audio_path}")
    with open(base_json_path + "hindi_dub.json", "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)


def debug_single_tts(sub, speaker_wav, output_path="debug_output.wav"):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to("cuda")
    text = sub["text"]
    
    if not text.strip():
        print("⚠️ Subtitle text is empty, skipping.")
        return

    print(f"[DEBUG] Generating audio for: {text}")

    print(f"[DEBUG] Raw text repr: {repr(text)}")

    # Run TTS
    raw_output = tts.tts(text=text, speaker_wav=speaker_wav, language="hi")

    # Inspect raw output
    print(f"[RAW OUTPUT TYPE] {type(raw_output)}")

    # Handle different output formats
    if isinstance(raw_output, list):
        print(f"[LIST] Length: {len(raw_output)} | Type of first element: {type(raw_output[0])}")
        
        # If it's a list of floats (not a list of arrays)
        if isinstance(raw_output[0], np.float32) or isinstance(raw_output[0], float):
            raw_output = np.array(raw_output, dtype=np.float32)
            print(f"[LIST Converted to ndarray] Shape: {raw_output.shape}")
        else:
            print(f"[LIST First array] Shape: {raw_output[0].shape}")
            raw_output = raw_output[0]  # Assume first is the actual audio

    else:
        print(f"[ARRAY] Shape: {raw_output.shape}")

    print(f"[First 10 samples]: {raw_output[:10]}")

    # Convert float32 audio (-1.0 to 1.0) to int16 PCM
    wav_array = (raw_output * 32767).astype("int16")

    debug_clip = AudioSegment(
        wav_array.tobytes(),
        frame_rate=tts.synthesizer.output_sample_rate,
        sample_width=2,
        channels=1
    )


    debug_clip.export(output_path, format="wav")
    print(f"✅ Debug audio exported: {output_path}")


def is_speakable(text: str) -> bool:
    """
    Returns True if the text contains actual speech content (Hindi/English letters),
    and is not just filler characters like '.', '-', '*', etc.
    """
    text = text.strip()
    
    # Reject if only punctuation or repeated characters
    if re.fullmatch(r'[\.\-\*\s\?!…‥、।,~#@%^&+=<>|/\\`\'"]+', text):
        return False

    # Reject if it’s a long repeat of same non-word char (like '........' or '---')
    if len(set(text)) == 1 and not text.isalnum():
        return False

    # Accept if it contains Hindi, Urdu, English, or meaningful text
    return bool(re.search(r'[a-zA-Z\u0900-\u097F]', text))
