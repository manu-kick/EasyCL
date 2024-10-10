import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from gensim.models import Word2Vec
from torchvision import models
import librosa
import numpy as np
from loss import * 
from models import TextEncoder,AudioEncoder,VisionEncoder
from pipelines import *
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
import torchaudio
import os
import numpy as np
from torchaudio.transforms import Resample
import pandas as pd
from tqdm.auto import tqdm
import wave
import random
from torch import nn,Tensor,optim
from sklearn.model_selection import GroupShuffleSplit
from dataloader import *
from torchaudio import transforms as T
import wandb



def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed=42
seed_everything(seed)
wandb.init(
        # set the wandb project where this run will be logged
        project ="ProveContrastiveLearning",

        # track hyperparameters and run metadata
        config={
        "desc":"Finetuning MSRVTT prima prova",
        "name": 'PrimaProvaLossVolume',  
        }
)



# Define your input dimensions and encoders
embedding_dim = 3  # Set to 3 for 3D latent space
output_dim = 3     # Embedding dimension of 3 for visualization (latent space)
w2v_path = './GoogleNews-vectors-negative300.bin'

#Define Encoders
text_encoder = TextEncoder(word2vec_model_path=w2v_path,embedding_size=embedding_dim).to('cuda')
audio_encoder = AudioEncoder( input_channels=1, output_dim=output_dim).to('cuda')
vision_encoder = VisionEncoder(input_channels=1, output_dim=output_dim).to('cuda')
contra_temp = nn.Parameter(torch.tensor(0.07))

# Create optimizer and criterion
optimizer = torch.optim.Adam(list(text_encoder.parameters()) + 
                              list(audio_encoder.parameters()) + 
                              list(vision_encoder.parameters()) + [contra_temp], lr=1e-4)


# Define the dataset roots
# MNIST will be downloaded with create_dataset function
# Download audioMNIST from kaggle https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data
mnist_root = './mnist_data'
audio_mnist_root = './mnist_data/audio-MNIST/data'

# Instantiate your dataset
mnist_transform = transforms.Compose([ transforms.ToTensor()])
audio_transform =  T.MelSpectrogram(sample_rate=48000,n_fft=512,n_mels=64) #Resample( new_freq=16000)  # Example transform: resample audio
metadata = get_metadata(audio_mnist_root)
mnist_audio_dataset_train, mnist_audio_dataset_test = create_datasets( metadata=metadata,
                                                                      test_size=0.0002,    
                                                                      mnist_root=mnist_root,
                                                                      audio_mnist_root=audio_mnist_root,
                                                                      mnist_transform=mnist_transform,
                                                                      audio_transform=audio_transform) 



# Create a DataLoader to load data in batches
unique_label_sampler_train = UniqueLabelSampler(mnist_audio_dataset_train, batch_size=10)
dataloader_train = DataLoader(mnist_audio_dataset_train,batch_size=10,sampler=unique_label_sampler_train)
dataloader_test = DataLoader(mnist_audio_dataset_test, batch_size=10, shuffle=False)
print(f'Train Dataloader len = {len(dataloader_train)}')
print(f'Test Dataloader len = {len(dataloader_test)}')


# Train and visualize the latent space
train_model_with_visualization(text_encoder, audio_encoder, vision_encoder, dataloader_train, dataloader_test,optimizer, device='cuda', num_iterations=10000,contra_temp=contra_temp)
