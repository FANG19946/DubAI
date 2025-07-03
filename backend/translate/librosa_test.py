import librosa
import soundfile as sf

# Load audio file (mono)
y, sr = librosa.load("test_sample2.wav")  # Load without resampling

# Time-stretch: speed up (2x)
y_fast = librosa.effects.time_stretch(y, rate=2.0)
sf.write("test_fast.wav", y_fast, sr)

# Time-stretch: slow down (0.5x)
y_slow = librosa.effects.time_stretch(y, rate=0.5)
sf.write("test_slow.wav", y_slow, sr)
