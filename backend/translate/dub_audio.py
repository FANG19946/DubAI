import torch
from TTS.api import TTS
from pydub import AudioSegment


device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_audio(subtitles, speaker_wav, input_audio_path, output_audio_path):
    """
    Replaces segments in input audio using XTTS-dubbed subtitles.
    """
    tts = TTS(model_name = "tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)

    original = AudioSegment.from_wav(input_audio_path)
    frame_rate = original.frame_rate
    audio_duration = len(original)

    for i, sub in enumerate(subtitles):
        text = sub["text"]
        start = sub["start"]
        end = sub["end"]
        duration = end - start

        end = min(end, audio_duration)
        if start >= audio_duration:
            break

        if duration <= 0 or not text.strip():
            print(f"⚠️ Skipping invalid subtitle at index {sub.get('index', i)}")
            continue

        # Generate audio with XTTS
        wav_array = tts.tts(text=text, speaker_wav=speaker_wav, language="hi")

        # 🛠 Handle list output
        if isinstance(wav_array, list):
            wav_array = wav_array[0]

        # 🔁 Convert from float32 (-1.0 to 1.0) to int16 range (-32768 to 32767)
        wav_array = (wav_array * 32767).astype("int16")

        dubbed = AudioSegment(
            wav_array.tobytes(),
            frame_rate=tts.synthesizer.output_sample_rate,
            sample_width=2,
            channels=1
        ).set_frame_rate(frame_rate)

        dubbed = adjust_to_duration(dubbed, duration)

        # Replace segment in original
        original = original[:start] + dubbed + original[end:]

    original.export(output_audio_path, format="wav")
    print(f"✅ Dubbed audio exported to: {output_audio_path}")



def adjust_to_duration(audio, target_duration):
    """
    Adjust audio length by speeding up or padding to match target duration in ms.
    """
    current_duration = len(audio)
    if current_duration > target_duration:
        speed_factor = current_duration / target_duration
        audio = audio.speedup(playback_speed=speed_factor, chunk_size=50, crossfade=25)
    elif current_duration < target_duration:
        silence = AudioSegment.silent(duration=target_duration - current_duration)
        audio += silence
    return audio