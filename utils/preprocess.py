import numpy as np
import matplotlib.pyplot as plt

import librosa
from librosa import display

import parselmouth
from IPython.display import Audio

from .object import AudioObject
    
def load_audio(file_path, sampling_rate = 16000):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    snd = parselmouth.Sound(file_path)
    if snd.sampling_frequency != sampling_rate:
        snd = snd.resample(sampling_rate)
    return AudioObject(y, sr, snd)

def display_waveform(audio:AudioObject):
    plt.figure(figsize=(15, 6))

    # Display waveform using librosa
    plt.subplot(2, 1, 1)
    plt.title("Waveform using librosa")
    librosa.display.waveshow(audio.y, sr=audio.sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Display waveform using parselmouth
    plt.subplot(2, 1, 2)
    plt.title("Waveform using parselmouth")
    plt.plot(audio.snd.xs(), audio.snd.values.T)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def play_audio(audio, from_librosa=True):
    if from_librosa:
      print("Playing audio using librosa:")
      return Audio(data=audio.y, rate=audio.sr)
    else:    
      # Convert parselmouth.Sound to numpy array for playback
      snd_array = audio.snd.values.T.flatten()
      print("Playing audio using parselmouth:")
      return Audio(data=snd_array, rate=int(audio.snd.sampling_frequency))

def audio_strip(audio:AudioObject):
    y_stripped, idx = librosa.effects.trim(audio.y)
    start_time = idx[0] / audio.sr
    end_time = idx[1] / audio.sr
    stripped_sound = audio.snd.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
    return AudioObject(y=y_stripped, sr=audio.sr, snd=stripped_sound)