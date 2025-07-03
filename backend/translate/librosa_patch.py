import json
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import io

# 📥 Inputs
input_wav = "translated_audio.wav"
subtitle_json = "output_metadata.json"
output_wav = "dubbed_patched.wav"

# 🎧 Load full dubbed audio
print(f"📂 Loading: {input_wav}")
full_audio = AudioSegment.from_wav(input_wav)

# 📑 Load subtitle metadata
with open(subtitle_json, "r", encoding="utf-8") as f:
    subtitles = json.load(f)

patched_audio = full_audio
offset = 0  # Tracks cumulative stretch offset

for i, sub in enumerate(subtitles):
    try:
        start = sub["dub_start"] + offset
        end = sub["dub_end"] + offset
        target_duration = sub["end"] - sub["start"]

        segment = patched_audio[start:end]

        if abs(len(segment) - target_duration) < 30:
            print(f"✅ [# {i}] Skipping — duration close enough.")
            continue

        print(f"🎛️ [# {i}] Patching duration mismatch")
        print(f"    → Segment: {len(segment)}ms | Target: {target_duration}ms")

        # Convert to NumPy
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        sr = segment.frame_rate

        # Time-stretch
        ratio = len(segment) / target_duration
        stretched = librosa.effects.time_stretch(samples, rate=ratio)

        # Convert back to AudioSegment
        buf = io.BytesIO()
        sf.write(buf, stretched, sr, format="WAV")
        buf.seek(0)
        adjusted = AudioSegment.from_file(buf, format="wav")

        # Final pad/trim
        if len(adjusted) < target_duration:
            adjusted += AudioSegment.silent(duration=target_duration - len(adjusted))
        elif len(adjusted) > target_duration:
            adjusted = adjusted[:target_duration]

        # Replace in patched_audio
        patched_audio = patched_audio[:start] + adjusted + patched_audio[end:]

        # Track stretch offset
        offset += len(adjusted) - (end - start)

    except Exception as e:
        print(f"❌ [# {i}] Failed to patch: {e}")
        continue

# 💾 Save final result
patched_audio.export(output_wav, format="wav")
print(f"✅ Patched audio saved to: {output_wav}")
