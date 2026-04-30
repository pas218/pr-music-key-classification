# import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as pltimport
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import soundfile as sf
from torchvision.transforms import v2
from CNN_LSTM import CNN_LSTM
from PIL import Image
from key_classfn_utilities import *
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# WAV_FILENAME = "recording.wav"
# SPECT_FILENAME = "spect.png"
MODEL_FILENAME = "models/model_86acc_300e.pt"
TEST_SPECT_PATH = "spectrogram_outputs"
spect_dir = Path(TEST_SPECT_PATH)

# Configuration
# freq = 44100  # Sampling frequency
# duration = 20  # Recording duration in seconds

# print("Recording...")
# # Start recording
# recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)

# # Wait for the recording to finish
# sd.wait()
# print("Recording complete.")

# # Save as WAV file
# write(WAV_FILENAME, freq, recording)

# waveform_np, sample_rate = sf.read(WAV_FILENAME)
# waveform = torch.tensor(waveform_np, dtype=torch.float32)
# if waveform.ndim == 2:
#     waveform = waveform.T
# elif waveform.ndim == 1:
#     waveform = waveform.unsqueeze(0)

# waveform = waveform.mean(dim=0, keepdim=True)

# waveform = waveform.to(device)
# mel_transform = T.MelSpectrogram(
#     sample_rate=sample_rate,
#     n_fft=2096,
#     hop_length=512,
#     n_mels=128
# ).to(device)
# mel_spec = mel_transform(waveform)
# to_db = T.AmplitudeToDB().to(device)
# mel_spec_db = to_db(mel_spec)
# mel_spec_db_cpu = mel_spec_db.cpu().squeeze().numpy()
# plt.imshow(mel_spec_db_cpu, aspect='auto', origin='lower', cmap='gray')
# plt.tight_layout()
# plt.axis('off')
# plt.savefig(SPECT_FILENAME, bbox_inches='tight', transparent=True, pad_inches=0.0)
# plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = torch.load(MODEL_FILENAME, map_location=device, weights_only=False)
model.eval()

transform = v2.Compose([
    v2.ToImage(),                              # convert PIL → tensor first
    v2.Grayscale(num_output_channels=1),
    v2.Resize((450, 600), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

print("hi10")
for spect in spect_dir.glob('*.png'):
    # print(spect)
    img = Image.open(spect).convert("RGB")  # ensure 3 channels
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    #print("hi11")
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    #print("hi12")
    print(f'The model prediction for {spect} is {labels_map[predicted_class]}.')

