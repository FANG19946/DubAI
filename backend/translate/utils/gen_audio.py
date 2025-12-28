from parse_srt import parse_srt
from dub_audio import debug_single_tts, generate_audio  # assuming the new function lives there
from speaker_utils import export_speaker_cache, gen_dialogue_array
def main():

    base_srt_path = "data/srt/"  
    base_wav_path = "data/wav/"  

    audio_file = "eng_10_to_15.wav"
    srt_file = "hi_10_to_15.srt"
    srt_path = base_srt_path + srt_file
    input_audio = base_wav_path + audio_file
    output_audio = base_wav_path + "hi_10_to_15.wav"
    segments = gen_dialogue_array(audio_file, srt_file)
    speaker_map = export_speaker_cache(segments)

    # Step 1: Parse subtitle file
    subtitles = parse_srt(srt_path)

    # Step 2: Generate dubbed audio
    generate_audio(
        subtitles=subtitles,
        speaker_map=speaker_map,
        input_audio_path=input_audio,
        output_audio_path=output_audio
    )

    # debug_single_tts(subtitles[0], speaker_wav="clean_voice.wav", output_path="debug_sub0.wav")


if __name__ == "__main__":
    main()
