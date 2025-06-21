from parse_srt import parse_srt, write_srt_from_parsed
from translate_utils import translate_subtitles_to_hindi

# Step 1: Parse original English subtitles
parsed = parse_srt("test.srt")  # or your actual path

# Step 2: Translate the 'text' fields to Hindi
translated = translate_subtitles_to_hindi(parsed)

# Step 3: Write translated subtitles to a new file
write_srt_from_parsed(translated, "translated_output.srt")
