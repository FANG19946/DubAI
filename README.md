# A tool that can dub videos using AI

- Generate subtitles .srt file from video using Whisper
- Translate subtitles using IndicTrans2
- TTS using Coqui

This is all for now

I am coming back to this after a long gap so this is me trying to piece everything back together

### dub_audio/generate_audio
This function seems to be the main part that generates the hindi audio using xtts from the translated hindi SRT and at the end it also makes a hindi_dub.json which has things like:

      {
      "index": 1,
      "timestamp": "00:00:00,000 --> 00:00:12,380",
      "text": "नॉर्मन, तुम ठीक हो जा रहे हो।",
      "start": 0,
      "end": 12380,
      "dub_start": 0,
      "dub_end": 8118
      },
Container Name:


### gen_audio.py
This is more like a runner or captain function that calls functions from all other parts
It calls parse_srt which basically just parses the srt file and returns an array which has important things like:

    "index": index,
    "timestamp": timestamp,
    "text": text,
    "start": start_ms,
    "end": end_ms
next it calls the generate_audio to generate the output audio and json 
Container Name:
***Important for running, Useless for debugging***


*`Another thing I should add is all of these run inside their own docker containers, a better and much scalable way would be to replace these with venvs which I should work on but I guess that can be delegated to a lesser productive day
Also this code should be cleaned up because currently it is a very unstructured and disorganized and can only be understood by me (partly)`*



### gen_sub.py
This is another engine or captain function like gen_audio it basically calls all the different functions required for generating the hindi SRT file from the English SRT file
***Important for running, Useless for debugging***

### librosa_patch.py
This file basically matches the generated audio segments to the length of the original english audio segments

*`I think later on I can refactor this to be done as the audio segments are generated to increase parallelism but that is not an immediate concern for now at all`*

### speaker_utils.py
This was the last thing I was working on trying to get speaker diarization and mapping and coming up with way for speaker diarization
Now that I have worked a little more on this I have realised its really difficult to rely on Global similarity in embeddings and currently I am using an approach where I find segments in the neighborhood of the segment I need more samples of and then find similarity between them and select the top 2 ones, it seems like a good choice to expand the speaker audio.
Now for the next part in this is to actually expand the speaker recordings in a BFS like fashion and to anchor it more towards the original audio we can use a weight like alpha when we use it with the other recordings which essentially becomes a hyperparameter

### test_gen_diarization
This is the captain function for the diarization part

### translate_utils.py
This is the heart of the IndicTrans2 English SRT to Hindi SRT generator and hold the functions with the core functionality
