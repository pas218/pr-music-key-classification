import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as pltimport
import torch
import torchaudio
import torchaudio.transforms as T
from CNN_LSTM import CNN_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAV_FILENAME = "recording.wav"
SPECT_FILENAME = "spect.png"
WEIGHTS_FILENAME = "weights.pt2"

# Configuration
freq = 44100  # Sampling frequency
duration = 5  # Recording duration in seconds

print("Recording...")
# Start recording
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)

# Wait for the recording to finish
sd.wait()
print("Recording complete.")

# Save as WAV file
write(WAV_FILENAME, freq, recording)

waveform_np, sample_rate = sf.read(WAV_FILENAME)
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

plt.imshow(mel_spec_db_cpu, aspect='auto', origin='lower', cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.savefig(SPECT_FILENAME, bbox_inches='tight', transparent=True, pad_inches=0.0)
plt.close()


model = CNN_LSTM()
model.load_state_dict(WEIGHTS_FILENAME)

model.eval()

output = model(SPECT_FILENAME)

print(f'The model prediction is {output}.')


