import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf

SOUND_OUTPUTS_PATH = "sound_outputs"
SPECTROGRAM_OUTPUTS_PATH = "spectrogram_outputs"
device = torch.device("cuda")

sound_directory = Path(SOUND_OUTPUTS_PATH)

for sound_file in sound_directory.glob('*.wav'):
    spect_filename = sound_file.name.replace("output", "spect").replace(".wav", "")
    
    # waveform, sample_rate = torchaudio.load(f"{SOUND_OUTPUTS_PATH}/{sound_file.name}")
    waveform_np, sample_rate = sf.read(f"{SOUND_OUTPUTS_PATH}/{sound_file.name}")
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.T
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2096,
        hop_length=512,
        n_mels=128
    ).to(device)

    mel_spec = mel_transform(waveform)
    to_db = T.AmplitudeToDB().to(device)
    mel_spec_db = to_db(mel_spec)

    mel_spec_db_cpu = mel_spec_db.cpu().squeeze().numpy()

    plt.imshow(mel_spec_db_cpu, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"{SPECTROGRAM_OUTPUTS_PATH}/{spect_filename}.png", bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
