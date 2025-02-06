import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from loss import * 
from models import TextEncoder,AudioEncoder,VisionEncoder
from pipelines import *
import torch
from torch.utils.data import  DataLoader
from torchvision import transforms
import os
import numpy as np
import random
from torch import nn
from dataloader import *
from torchaudio import transforms as T
import wandb
from config import *

cf = Config()


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(cf.seed)

if cf.wandb == True:
    wandb.init(
            settings=wandb.Settings(init_timeout=120),
            # set the wandb project where this run will be logged
            project ='prova', #"MNIST_ContrastiveLearning",
            #name = cf.run,
            #id= cf.run,
            # track hyperparameters and run metadata
            config= cf.log_config()
    )



# Define your input dimensions and encoders
embedding_dim = 3  # Set to 3 for 3D latent space
output_dim = 3     # Embedding dimension of 3 for visualization (latent space)
w2v_path = './GoogleNews-vectors-negative300.bin'

#Define Encoders
text_encoder = TextEncoder(word2vec_model_path=w2v_path,embedding_size=embedding_dim).to('cuda')
audio_encoder = AudioEncoder( input_channels=1, output_dim=output_dim).to('cuda')
vision_encoder = VisionEncoder(input_channels=1, output_dim=output_dim).to('cuda')
contra_temp = nn.Parameter(torch.tensor(cf.contra_temp_init))

# Create optimizer and criterion
optimizer = torch.optim.Adam(list(text_encoder.parameters()) + 
                              list(audio_encoder.parameters()) + 
                              list(vision_encoder.parameters()) + [contra_temp], lr=cf.lr)

# Create learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)


# Define the dataset roots
# MNIST will be downloaded with create_dataset function
# Download audioMNIST from kaggle https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data
mnist_root = './mnist_data'
audio_mnist_root = './mnist_data/audio-MNIST/data'

# Instantiate your dataset
mnist_transform = transforms.Compose([ transforms.ToTensor()])
audio_transform =  T.MelSpectrogram(sample_rate=cf.sample_rate,n_fft=cf.n_fft,n_mels=cf.n_mels) #Resample( new_freq=16000)  # Example transform: resample audio

#If exists load the dataset
if os.path.exists('mnist_audio_dataset_train.pt'):
    mnist_audio_dataset_train = torch.load('mnist_audio_dataset_train.pt')
    mnist_audio_dataset_test = torch.load('mnist_audio_dataset_test.pt')
else:
    metadata = get_metadata(audio_mnist_root)
    mnist_audio_dataset_train, mnist_audio_dataset_test = create_datasets( metadata=metadata,
                                                                          test_size=0.0002,    
                                                                          mnist_root=mnist_root,
                                                                          audio_mnist_root=audio_mnist_root,
                                                                          mnist_transform=mnist_transform,
                                                                          audio_transform=audio_transform) 
    # save the dataset to a file
    torch.save(mnist_audio_dataset_train, 'mnist_audio_dataset_train.pt')
    torch.save(mnist_audio_dataset_test, 'mnist_audio_dataset_test.pt')


# Create a DataLoader to load data in batches
unique_label_sampler_train = UniqueLabelSampler(mnist_audio_dataset_train, batch_size=cf.batch_size)
dataloader_train = DataLoader(mnist_audio_dataset_train,batch_size=cf.batch_size,sampler=unique_label_sampler_train)
dataloader_test = DataLoader(mnist_audio_dataset_test, batch_size=cf.batch_size, shuffle=False)
print(f'Train Dataloader len = {len(dataloader_train)}')
print(f'Test Dataloader len = {len(dataloader_test)}')


# Train and visualize the latent space
train_model_with_visualization(cf,text_encoder, audio_encoder, vision_encoder, dataloader_train, dataloader_test,optimizer, device=cf.device, num_iterations=cf.num_iterations,contra_temp=contra_temp)
