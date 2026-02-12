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


# Dictionary for textual descriptions of digits
digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

def get_metadata(root : str) -> pd.DataFrame:

    metadata = {
        'path' : [],
        'label' : [],
        'person_id' : [],
        'recording_id' : [],
        'audio_length' : [],
        'sr' : []
    }

    subfolders = os.listdir(root)

    for subfolder in subfolders:
            
        subfolder_path = os.path.join(root, subfolder)

        if os.path.isdir(subfolder_path):
                
            audios = os.listdir(subfolder_path)

            for audio in tqdm(audios, desc=f"Processing {subfolder_path}"):

                audio_path = os.path.abspath(os.path.join(subfolder_path, audio))
                name,_ = os.path.splitext(audio)
                label,person_id,recording_id = name.split('_')
                
                metadata['path'].append(audio_path)
                metadata['label'].append(int(label))
                metadata['person_id'].append(int(person_id))
                metadata['recording_id'].append(int(recording_id))
                
                with wave.open(audio_path, 'r') as f:

                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    
                    metadata['audio_length'].append(duration)
                    metadata['sr'].append(rate)

    return pd.DataFrame(metadata) 

# Define your custom dataset that combines MNIST and Audio-MNIST
class MNIST_AudioDataset(Dataset):
    def __init__(self, mnist_root, audio_mnist_root, mnist_transform=None, audio_transform=None,split='train',metadata=None):
        """
        Args:
            mnist_root (str): Path to the MNIST dataset.
            audio_mnist_root (str): Path to the Audio-MNIST dataset.
            transform (callable, optional): Optional transform to be applied on MNIST image.
            audio_transform (callable, optional): Optional transform to be applied on Audio-MNIST audio.
        """
        # Load MNIST dataset
        self.split=split
        if split=='train':
            self.train=True
        else:
            self.train=False
        self.mnist_data = datasets.MNIST(mnist_root, train=self.train, download=True, transform=mnist_transform)
        self.metadata = metadata #get_metadata(audio_mnist_root)
        self.seq_len  = 48000

        if self.split is not None:
            self.metadata = self.metadata[self.metadata['split'] == self.split]

    
        self.transform = mnist_transform
        self.audio_transform = audio_transform

    def _adjust_audio_length(self,audio_data : Tensor) -> Tensor:

        C,L = audio_data.shape

        if L > self.seq_len:
            audio_data = audio_data[:,:self.seq_len]
        elif L < self.seq_len:
            pad = torch.zeros(C,self.seq_len - L)
            audio_data = torch.cat([audio_data,pad],dim=-1)

        return audio_data

    def __len__(self):
        return min(len(self.mnist_data), len(self.metadata))  # Ensure both datasets are aligned

    def __getitem__(self, idx):
        # Get MNIST image and label
        #print(idx)
        mnist_img, label = self.mnist_data[idx]

        list_audios_with_label= self.metadata[self.metadata["label"]==int(label)]['path']
        list_audios_with_label=list(list_audios_with_label)

        path=random.choice(list_audios_with_label)
        path = os.path.normpath(path)
        # Get Audio-MNIST audio for the same label (assuming audio corresponds to the same label)
        waveform, sample_rate = torchaudio.load(path,normalize=True,channels_first=True)
        waveform = self._adjust_audio_length(waveform)

        spectogram=self.audio_transform(waveform)

        # Get textual description of the digit
        text_description = digit_to_text[label]
        

        
        return mnist_img, spectogram, text_description, label
    
def create_datasets(
    metadata : pd.DataFrame,
    test_size : float = 0.2,
    mnist_root=None,
    audio_mnist_root=None,
    mnist_transform=None,
    audio_transform=None
) -> tuple[MNIST_AudioDataset,MNIST_AudioDataset]:

    # gss = GroupShuffleSplit(n_splits=1,test_size=test_size)
    # _,test_indices = next(iter(gss.split(metadata,groups=metadata['person_id'])))

    # metadata['split'] = 'train'
    # metadata.loc[test_indices,'split'] = 'test'

    test_size = 200

    # Create a 'split' column and set all entries to 'train' initially
    metadata['split'] = 'train'

    # Randomly sample 100 unique indices
    test_indices = np.random.choice(metadata.index, size=test_size, replace=False)

    # Mark the selected indices as 'test'
    metadata.loc[test_indices, 'split'] = 'test'

    train_data = MNIST_AudioDataset(mnist_root, audio_mnist_root, mnist_transform=mnist_transform, audio_transform=audio_transform,split="train",metadata=metadata)
    test_data = MNIST_AudioDataset(mnist_root, audio_mnist_root, mnist_transform=mnist_transform, audio_transform=audio_transform,split="test",metadata=metadata)

    return train_data,test_data




class UniqueLabelSampler(Sampler):
    # def __init__(self, dataset, batch_size):

    #     self.dataset = dataset
    #     self.batch_size = batch_size

    #     # Extract the labels (digits) from the dataset
    #     self.labels = np.array([dataset[i][3] for i in range(len(dataset))])  # dataset[i][1] is the label
        
    #     # Create a dictionary that maps each label (digit) to its indices in the dataset
    #     self.label_to_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # MNIST labels without triggering __getitem__()
        self.labels = dataset.mnist_data.targets.cpu().numpy()

        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }

    def __iter__(self):
        """
        Sample indices for the batch. Each batch will have unique labels (digits).
        """
        while True:
            indices = []
            chosen_labels = np.random.choice(
                np.unique(self.labels), size=self.batch_size, replace=False
            )  # Sample batch_size unique labels
            for label in chosen_labels:
                # Select a random sample for each label (digit)
                indices.append(np.random.choice(self.label_to_indices[label]))
            
            for idx in indices:
                yield idx  # Yield a single index at a time, so the DataLoader can fetch the item

    def __len__(self):
        """
        Return the number of batches that can be made from the dataset.
        """
        return len(self.dataset)  # This returns 10, the number of unique labels (0-9)



# HOW TO USE IT

# def seed_everything(seed: int):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# seed=40
# seed_everything(seed)

# mnist_transform = transforms.Compose([ transforms.ToTensor()])
# audio_transform = None #Resample( new_freq=16000)  # Example transform: resample audio

# # Define the dataset roots
# # MNIST will be downloaded with create_dataset function
# # Download audioMNIST from kaggle https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data

# mnist_root = './mnist_data'
# audio_mnist_root = './mnist_data/audio-MNIST/data'

# # Instantiate your dataset
# metadata = get_metadata(audio_mnist_root)
# mnist_audio_dataset_train, mnist_audio_dataset_test = create_datasets( metadata=metadata,test_size=0.1,mnist_transform=mnist_transform,audio_transform=audio_transform) 


# unique_label_sampler_train = UniqueLabelSampler(mnist_audio_dataset_train, batch_size=10)


# # Create a DataLoader to load data in batches
# dataloader_train = DataLoader(mnist_audio_dataset_train, batch_size=10, sampler=unique_label_sampler_train)
# dataloader_test = DataLoader(mnist_audio_dataset_test, batch_size=10, shuffle=False)


# print(len(dataloader_train))
# print(len(dataloader_test))
# # Example of iterating through the dataloader
# for batch_idx, (mnist_img, waveform, text_desc, label) in enumerate(dataloader_train):
#     print(f"Batch {batch_idx+1}")
#     print(f"MNIST Image Shape: {mnist_img.shape}")
#     print(f"Audio Waveform Shape: {waveform.shape}")
#     print(f"Text Description: {text_desc}")
#     print(f"Labels: {label}")
#     break
