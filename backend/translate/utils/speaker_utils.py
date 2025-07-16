from pydub import AudioSegment
from parse_srt import parse_srt

def gen_dialogue_wav(audio_path, silence_duration=1000, srt_path = 'eng_10_to_15.srt' ,save_path = 'dialogue_diarization.wav' ):
    
    base_wav_path = "data/wav/"  
    base_srt_path = "data/srt/"  

    audio = AudioSegment.from_wav(base_wav_path + audio_path)
    subtitles = parse_srt(base_srt_path + srt_path)
    silence = AudioSegment.silent(silence_duration)
    dialogue = silence

    


    for sub in subtitles:
        begin = sub["start"]
        end = sub["end"]
        segment = audio[begin:end]
        dialogue = dialogue + silence + segment

    dialogue.export(base_wav_path + save_path, format="wav")



