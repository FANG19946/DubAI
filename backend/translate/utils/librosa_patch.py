import json
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import io

def patch_audio_with_librosa():

    base_wav_path = "data/wav/"  
    base_json_path = "data/json/"  



    input_wav =  base_wav_path + "hi_10_to_15.wav"
    output_wav =base_wav_path + "librosa_patched.wav"
    subtitle_json = base_json_path + "output_metadata.json"

    print(f"📂 Loading: {input_wav}")
    full_audio = AudioSegment.from_wav(input_wav)

    with open(subtitle_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    patched_audio = full_audio
    offset = 0

    for i, sub in enumerate(subtitles):
        try:
            if "dub_start" not in sub or "dub_end" not in sub:
                print(f"⚠️ [# {i}] Skipping: no dub_start/dub_end")
                continue

            start = sub["dub_start"] + offset
            end = sub["dub_end"] + offset
            target_duration = sub["end"] - sub["start"]

            if end <= start:
                print(f"❌ [# {i}] Invalid segment range ({start}-{end})")
                continue

            segment = patched_audio[start:end]

            if len(segment) < 300:
                print(f"⏭️ [# {i}] Skipping short segment ({len(segment)}ms)")
                continue

            if abs(len(segment) - target_duration) < 30:
                print(f"✅ [# {i}] Duration close enough — skipping.")
                continue

            print(f"🎛️ [# {i}] Patching: {len(segment)}ms → {target_duration}ms")

            segment = segment.set_channels(1).set_frame_rate(22050)
            samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
            sr = segment.frame_rate

            ratio = len(segment) / target_duration
            print(f"📐 Stretch ratio: {ratio:.3f}")

            stretched = librosa.effects.time_stretch(samples, rate=ratio)

            buf = io.BytesIO()
            sf.write(buf, stretched, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            adjusted = AudioSegment.from_file(buf, format="wav")

            if len(adjusted) < target_duration:
                adjusted += AudioSegment.silent(duration=target_duration - len(adjusted))
            elif len(adjusted) > target_duration:
                adjusted = adjusted[:target_duration]

            adjusted = adjusted.fade_in(10).fade_out(10)

            patched_audio = patched_audio[:start] + adjusted + patched_audio[end:]
            print(f"📏 [# {i}] Adjusted length: {len(adjusted)}ms, Target: {target_duration}ms, Delta: {len(adjusted) - target_duration}ms")

            offset += len(adjusted) - (end - start)
            print(f"🧭 [# {i}] Offset updated: {offset}ms\n")

        except Exception as e:
            print(f"❌ [# {i}] Error: {e}")
            continue

    patched_audio.export(output_wav, format="wav")
    print(f"✅ Patched audio saved: {output_wav}")


# 🔁 Run it immediately
patch_audio_with_librosa()
