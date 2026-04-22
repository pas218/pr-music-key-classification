import sounddevice as sd
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAV_FILENAME = "recording.wav"
SPECT_FILENAME = "spect.png"
WEIGHTS_FILENAME = ".\weights_04_20.pth"
#MODEL_FILENAME = "./model_weights_83acc.pth"

# Configuration
freq = 44100  # Sampling frequency
duration = 20  # Recording duration in seconds

print("Recording...")
# Start recording
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)

# Wait for the recording to finish
sd.wait()
print("Recording complete.")

# Save as WAV file
write(WAV_FILENAME, freq, recording)

#print("hi1")
waveform_np, sample_rate = sf.read(WAV_FILENAME)
waveform = torch.tensor(waveform_np, dtype=torch.float32)
if waveform.ndim == 2:
    waveform = waveform.T
elif waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)

#print("hi2")
waveform = waveform.mean(dim=0, keepdim=True)

#print("hi3")
waveform = waveform.to(device)
#print("hi4")
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2096,
    hop_length=512,
    n_mels=128
).to(device)
#print("hi5")
mel_spec = mel_transform(waveform)
to_db = T.AmplitudeToDB().to(device)
mel_spec_db = to_db(mel_spec)
#print("hi6")
mel_spec_db_cpu = mel_spec_db.cpu().squeeze().numpy()
#print("hi7")
plt.imshow(mel_spec_db_cpu, aspect='auto', origin='lower', cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.savefig(SPECT_FILENAME, bbox_inches='tight', transparent=True, pad_inches=0.0)
plt.close()
#print("hi8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#print("hi8.1")
model = CNN_LSTM(num_classes=24, num_filters=16, num_hidden_units=128).to(device)
#print("hi8.2")

try:
    model.load_state_dict(torch.load(WEIGHTS_FILENAME, map_location=torch.device('cpu')))
except Exception as e:
    print(f"An error occurred: {e}")

#print("hi8.3")
##model = torch.load(MODEL_FILENAME)
#print(model)
#model = CNN_LSTM()
#model.load_state_dict(torch.load(WEIGHTS_FILENAME, weights_only=True))

#print("hi9")
model.eval()

transform = v2.Compose([
    v2.ToImage(),                              # convert PIL → tensor first
    v2.Grayscale(num_output_channels=1),
    v2.Resize((450, 600), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

#print("hi10")
img = Image.open(SPECT_FILENAME).convert("RGB")  # ensure 3 channels
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0).to(device)
#print("hi11")
# Run inference
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
#print("hi12")
print(f'The model prediction is {labels_map[predicted_class]}.')

