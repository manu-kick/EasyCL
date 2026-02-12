import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from gensim.models import Word2Vec
import gensim
from torchvision import models
import librosa
import numpy as np
from torch import nn,Tensor,optim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# -------------------
# Text Encoder (Word2Vec)
# -------------------

class TextEncoder(nn.Module):
    def __init__(self, word2vec_model_path, embedding_size=3):
        """
        Args:
            word2vec_model_path (str): Path to the pretrained Word2Vec model.
            embedding_size (int): Size of the sentence embedding.
        """
        super(TextEncoder, self).__init__()
        # Load pre-trained Word2Vec model
        kv_path = "./word2vec.kv"

        if os.path.exists(kv_path):
            # Fast: memory-mapped load
            self.word2vec = KeyedVectors.load(kv_path, mmap='r')
        else:
            # Slow: only happens once
            self.word2vec = KeyedVectors.load_word2vec_format(
                word2vec_model_path,
                binary=True
            )
            self.word2vec.save(kv_path)
            
        # if "word2vec.kv" file exists, use the following line instead:
        # self.word2vec = Word2Vec.load(word2vec_model_path, , mmap='r')
        
        self.embedding_size = embedding_size
        
        # We can add a linear layer to project the sentence embedding to the desired size.
        self.projection_layer = nn.Linear(self.word2vec.vector_size, embedding_size)

    def preprocess_text(self, text):
        """
        Preprocess the input text: tokenization and stopwords removal.
        
        Args:
            text (str): Input sentence to preprocess.
        
        Returns:
            list: List of preprocessed words.
        """
        # Tokenize the sentence
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords (optional, but often useful for text embeddings)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        
        return tokens

    def forward(self, text_batch):
        """
        Generate the sentence embeddings for a batch of texts using Word2Vec.
        
        Args:
            text_batch (tuple): Tuple of input sentences (list of strings).
        
        Returns:
            torch.Tensor: Batch of sentence embeddings with the specified embedding size.
        """
        batch_size = len(text_batch)
        all_sentence_embeddings = []
        
        # Process each sentence in the batch
        for text in text_batch:
            # Preprocess the text (tokenize, remove stopwords, etc.)
            words = self.preprocess_text(text)
            
            # Fetch word embeddings for each word in the sentence
            word_embeddings = []
            for word in words:
                if word in self.word2vec:
                    word_embeddings.append(self.word2vec[word])
                else:
                    # If the word is not in the vocab, use a zero vector
                    word_embeddings.append(np.zeros(self.word2vec.vector_size))
            
            # Average the word embeddings to get a fixed-size vector
            if len(word_embeddings) > 0:
                sentence_embedding = np.mean(word_embeddings, axis=0)
            else:
                # If the sentence was empty or contained no valid words, use a zero vector
                sentence_embedding = np.zeros(self.word2vec.vector_size)
            
            # Convert to tensor and project to desired embedding size
            sentence_embedding = torch.tensor(sentence_embedding, dtype=torch.float32)
            sentence_embedding = sentence_embedding.to('cuda')
            sentence_embedding = self.projection_layer(sentence_embedding)
            
            all_sentence_embeddings.append(sentence_embedding)
        
        # Stack the embeddings into a batch tensor
        batch_embeddings = torch.stack(all_sentence_embeddings, dim=0)
        
        return batch_embeddings
# --------------------
# Audio Encoder (Simple CNN)
# --------------------

class AudioEncoder(nn.Module):
    def __init__(self,input_channels, output_dim):
        super(AudioEncoder, self).__init__()
        
        # CNN Layer: Conv -> ReLU -> Pooling
        self.cnn_layers = nn.Sequential(
            # Input: [bs, 1, 64, 188]
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=1, padding=1), # Output: [bs, 32, 64, 188]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),               # Output: [bs, 32, 32, 94]
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), # Output: [bs, 64, 32, 94]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),               # Output: [bs, 64, 16, 47]
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), # Output: [bs, 128, 16, 47]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)               # Output: [bs, 128, 8, 23]
        )
        
        # Fully connected layers (MLP) after CNN
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                          # Flatten the output from CNN (size: [bs, 128*8*23])
            nn.Linear(128 * 8 * 23, 512),          # Fully connected layer 1 (Output: [bs, 512])
            nn.ReLU(),
            nn.Linear(512, output_dim)                      # Fully connected layer 2 (Output: [bs, 3])
        )

    def forward(self, x):
        # Pass through CNN layers for feature extraction
        x = self.cnn_layers(x)
        
        # Pass through fully connected layers to get embeddings
        x = self.fc_layers(x)
        
        return x

# --------------------
# Vision Encoder (Simple CNN)
# --------------------


class VisionEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(VisionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Update the input size of the fully connected layer to match the output shape of conv2
        self.fc1 = nn.Linear(64 * 7 * 7, output_dim)  # 64 channels, 8x8 spatial dimensions after pooling

    def forward(self, x):
        # Convolutional layers to process image data
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x


