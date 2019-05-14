from Wav2Letter.data import GoogleSpeechCommand
import torch
from Wav2Letter.decoder import GreedyDecoder
from scipy.io.wavfile import read
from sonopy import mfcc_spec
import numpy as np

def normalize(values):
    return (values - np.mean(values)) / np.std(values)

def wav2feat(wavPath):
    _,audio = read(wavPath)
    mfccs = mfcc_spec(
        audio, 16000, window_stride=(160, 80),
        fft_size=512, num_filt=20, num_coeffs=13
    )
    mfccs = normalize(mfccs)
    diff = 225 - mfccs.shape[0]
    mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")
    sample = torch.Tensor(mfccs)
    sample = sample.transpose(0, 1)
    return sample

def loadModel(modelPath):
    model = torch.load(modelPath,map_location='cpu')
    return model

def predInt2Letter(predTensor):
    charResult = []
    gs = GoogleSpeechCommand()
    for i in output.numpy():
        tmpletter = gs.intencode.index2char[i]
        charResult.append(tmpletter)
    return charResult

if __name__ == "__main__":
    wavPath = 'seven.wav'
    sample = wav2feat(wavPath)
    modelPath = './model/809.pth'
    model = loadModel(modelPath)
    log_probs = model.eval(sample)
    output = GreedyDecoder(log_probs)
    print("predicted Tensor", output)
    print("predicted Letter",predInt2Letter(output))