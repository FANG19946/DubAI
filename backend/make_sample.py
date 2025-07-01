from pydub import AudioSegment

# Load the original file
audio = AudioSegment.from_wav("sample.wav")

# Define start and end in milliseconds (e.g. 10 sec to 20 sec)
start_time = 0  # 10 seconds
end_time = 148720   # 20 seconds

# Slice the segment
extracted = audio[start_time:end_time]

# Save or use it directly
extracted.export("extracted_section.wav", format="wav")
