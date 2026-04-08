# Music Key Classification using CNN-LSTM
### ECE 2372: Pattern Recognition Final Project

Using the POP909 dataset, we leverage PyTorch to train a CNN-LSTM model to classify musical keys. We utilize Spatial Pyramid Pooling to make the model's processing agnostic to input lengths, allowing arbitrarily long songs to be classified (HOPEFULLY).

## Requirements
### FluidSynth
Converting from MIDI to WAV requires FluidSynth

This is an external install. Ensure CLI functionality works.

### sfArk
Converting from .sfArk to .sf2 requires sfArk (folder included)

This is (de)compression software so we can store soundfonts on Github