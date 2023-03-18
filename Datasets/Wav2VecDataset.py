"""
Feature extraction taken from https://bagustris.wordpress.com/2022/08/23/acoustic-feature-extraction-with-transformers/
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa


class Wav2VecDataset(Dataset):
    def __init__(self, filepath_list, label_list, wav2vecmodel="facebook/wav2vec2-base-960h", sampling_rate=16000):
        self.filepath_list = filepath_list
        self.label_list = label_list
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vecmodel)
        self.model = Wav2Vec2Model.from_pretrained(wav2vecmodel)
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.filepath_list)
    
    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        label = self.label_list[idx]

        array, fs = librosa.load(filepath, sr=self.sampling_rate) #torchaudio.load("Data\\cutfiles\\nips4b_birds_trainfile001_0.wav")
        input = self.processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

        # apply the model to the input array from wav
        with torch.no_grad():
            outputs = self.model(**input)

        # extract last hidden state, compute average, convert to numpy
        last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy().reshape((728,1))

        return last_hidden_states, label
