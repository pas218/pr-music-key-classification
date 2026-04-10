import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SOUND_OUTPUTS_PATH = "sound_outputs"
SPECTROGRAM_OUTPUTS_PATH = "spectrogram_outputs"

sound_directory = Path(SOUND_OUTPUTS_PATH)

for sound_file in sound_directory.glob('*.wav'):
    spect_filename = sound_file.name.replace("output", "spect").replace(".wav", "")

    y, sr = librosa.load(f"{SOUND_OUTPUTS_PATH}/{sound_file.name}" , sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(log_mel_spec, sr=sr)
    ax = plt.axes()
    ax.set_axis_off()
    plt.savefig(f"{SPECTROGRAM_OUTPUTS_PATH}/{spect_filename}.png", bbox_inches='tight', transparent=True, pad_inches=0.0)


    


