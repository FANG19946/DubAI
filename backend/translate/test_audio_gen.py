from parse_srt import parse_srt
from dub_audio import debug_single_tts, generate_audio  # assuming the new function lives there

def main():
    srt_path = "test_sample2_translated.srt"
    speaker_wav = "clean_voice.wav"
    input_audio = "test_sample2.wav"
    output_audio = "translated_audio.wav"

    # Step 1: Parse subtitle file
    subtitles = parse_srt(srt_path)

    # Step 2: Generate dubbed audio
    generate_audio(
        subtitles=subtitles,
        speaker_wav=speaker_wav,
        input_audio_path=input_audio,
        output_audio_path=output_audio
    )

    # debug_single_tts(subtitles[0], speaker_wav="clean_voice.wav", output_path="debug_sub0.wav")


if __name__ == "__main__":
    main()
