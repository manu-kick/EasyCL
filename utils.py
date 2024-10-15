
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from gensim.models import Word2Vec
from torchvision import models
import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loss import volume_computation3,volume_computation3Test,compute_centroidsTest,compute_centroids_only
from tqdm import tqdm
import wandb
from metrics import compute_metric_ret,compute_metric_ret2
import plotly.graph_objects as go



# THIS FUNCTION COMPUTE THE SIMILARITY AMONG DATA.
# IN THE MNIST SCENARIO WE CAN IMPOSE OUR OWN SEMANTIC
# E.G. NEAR NUMBERS->HIGHER SIMILARITY
def compute_similarity_matrix():
    # Define the number of items
    n_items = 10

    # Initialize an empty similarity matrix
    similarity_matrix = np.zeros((n_items, n_items))

    # Fill the matrix with similarity values iteratively
    for i in range(n_items):
        similarity_matrix[i, i] = 1.0  # Perfect similarity with itself

        # For items further away from item i (distance 1 to n_items - 1)
        for dist in range(1, n_items):
            # The target item index
            j = i + dist
            if j < n_items:
                # The similarity value decreases as distance increases
                similarity_value = 1.0 - dist * 0.2
                similarity_matrix[i, j] = similarity_value
                similarity_matrix[j, i] = similarity_value  # Ensure symmetry

    similarity_matrix = torch.tensor(similarity_matrix).to('cuda')
    # Print the similarity matrix
    print(similarity_matrix)

    #IF SMOOTHNING WITH SIGMOID STYLE
    similarity_matrix= 1-  (1/(1+torch.exp(-5*similarity_matrix)))

    #IF LINEAR DECADING
    #similarity_matrix = 1 - similarity_matrix
    print(similarity_matrix)
    return similarity_matrix



def visualize_3d(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):    
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-1, +1), ylim=(-1, +1))
    # Assign different markers and colors based on the labels
    unique_labels = np.unique(labels)
    
    # Colors for each label
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    
    # Plot text embeddings (stars)
    for i, label in enumerate(unique_labels):
        text_indices = np.where(np.array(labels) == label)[0]
        ax.scatter(text_embeddings[text_indices, 0], text_embeddings[text_indices, 1], text_embeddings[text_indices, 2],
                    label=f'Text - {label}', marker='*', color=colors(label), s=100)

    # Plot audio embeddings (triangles)
    for i, label in enumerate(unique_labels):
        audio_indices = np.where(np.array(labels) == label)[0]
        ax.scatter(audio_embeddings[audio_indices, 0], audio_embeddings[audio_indices, 1], audio_embeddings[audio_indices, 2],
                    label=f'Audio - {label}', marker='^', color=colors(label), s=100)

    # Plot vision embeddings (squares)
    for i, label in enumerate(unique_labels):
        vision_indices = np.where(np.array(labels) == label)[0]
        ax.scatter(vision_embeddings[vision_indices, 0], vision_embeddings[vision_indices, 1], vision_embeddings[vision_indices, 2],
                    label=f'Vision - {label}', marker='s', color=colors(label), s=100)

    # Labels and titl
    ax.set(xlim=(-1, +1), ylim=(-1, +1), zlim= (-1, +1))
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Latent Space Visualization of Text, Audio, and Vision Embeddings')
    
    # Add legend
    ax.legend()
    plt.savefig(f'latent space at {iterations}.png')
    


def visualize_3d_interactively(text_embeddings, audio_embeddings, vision_embeddings, iterations, labels):
    # Create a 3D plot
    fig = go.Figure()
    
    # Unique labels for coloring and grouping
    unique_labels = np.unique(labels)
    
    # Create a color map: we'll use Plotly's 'Viridis' color scale
    color_scale = 'Viridis'
    
    # Plot text embeddings (stars)
    for i, label in enumerate(unique_labels):
        text_indices = np.where(np.array(labels) == label)[0]
        fig.add_trace(go.Scatter3d(
            x=text_embeddings[text_indices, 0],
            y=text_embeddings[text_indices, 1],
            z=text_embeddings[text_indices, 2],
            mode='markers',
            marker=dict(
                symbol='x',
                size=5,
                color=i,  # Map the label index to a color
                colorscale=color_scale,  # Use the selected color scale
                colorbar=dict(title='Label Index')  # Color bar for reference
            ),
            name=f'Text - {label}'
        ))

    # Plot audio embeddings (triangles)
    for i, label in enumerate(unique_labels):
        audio_indices = np.where(np.array(labels) == label)[0]
        fig.add_trace(go.Scatter3d(
            x=audio_embeddings[audio_indices, 0],
            y=audio_embeddings[audio_indices, 1],
            z=audio_embeddings[audio_indices, 2],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=5,
                color=i,  # Map the label index to a color
                colorscale=color_scale,  # Use the selected color scale
                colorbar=dict(title='Label Index')  # Color bar for reference
            ),
            name=f'Audio - {label}'
        ))

    # Plot vision embeddings (squares)
    for i, label in enumerate(unique_labels):
        vision_indices = np.where(np.array(labels) == label)[0]
        fig.add_trace(go.Scatter3d(
            x=vision_embeddings[vision_indices, 0],
            y=vision_embeddings[vision_indices, 1],
            z=vision_embeddings[vision_indices, 2],
            mode='markers',
            marker=dict(
                symbol='square',
                size=5,
                color=i,  # Map the label index to a color
                colorscale=color_scale,  # Use the selected color scale
                colorbar=dict(title='Label Index')  # Color bar for reference
            ),
            name=f'Vision - {label}'
        ))

    # Set axis labels and title
    fig.update_layout(
        title=f'3D Latent Space Visualization of Text, Audio, and Vision Embeddings at Iteration {iterations}',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1, traceorder='normal')
    )

    # Show the interactive plot
    fig.show()

    # Optionally, save the plot to a static file
    fig.write_image(f'latent_space_at_{iterations}.png')
