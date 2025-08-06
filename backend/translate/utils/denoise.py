import noisereduce as nr
import librosa
import soundfile as sf

# Load the audio file
y, sr = librosa.load("input.wav", sr=None)

# Estimate noise from the first 0.5 sec (adjust if needed)
noise_sample = y[:int(sr * 0.5)]

# Perform noise reduction
reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

# Save output
sf.write("output_denoised.wav", reduced_noise, sr)
