import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from gensim.models import Word2Vec
from torchvision import models
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loss import volume_computation3,volume_computation3Test,compute_centroids
from tqdm import tqdm
import wandb
from metrics import compute_metric_ret,compute_metric_ret2



def eval(test_dataloader, text_encoder, audio_encoder, vision_encoder, device,iterations):
    """
    Extract embeddings from text, audio, and image modalities and visualize them in a 3D latent space.
    
    Args:
        test_dataloader (DataLoader): DataLoader containing the test data.
        text_encoder (nn.Module): Text encoder model.
        audio_encoder (nn.Module): Audio encoder model.
        vision_encoder (nn.Module): Vision encoder model.
        device (torch.device): The device to run the models on (CPU or GPU).
    """
    text_encoder = text_encoder.eval()
    audio_encoder = audio_encoder.eval()
    vision_encoder = vision_encoder.eval()
    # Lists to store embeddings and corresponding labels
    text_embeddings = []
    audio_embeddings = []
    vision_embeddings = []
    labels = []
    
    # Process all batches in the test dataloader
    with torch.no_grad():
        for batch_idx, (mnist_img, spectogram, text_description, label) in tqdm(enumerate(test_dataloader)):
            # Move inputs and labels to device
            mnist_img = mnist_img.to(device)
            spectogram = spectogram.to(device)
            text_description = text_description
            label = label.to(device)
            
            # Extract embeddings for each modality
            text_emb = text_encoder(text_description)
            audio_emb = audio_encoder(spectogram)
            vision_emb = vision_encoder(mnist_img)

            text_emb = F.normalize(text_emb,dim=-1)

            #print(text_embedding)
            audio_emb = F.normalize(audio_emb,dim=-1)
            #print(audio_embedding)
            vision_emb = F.normalize(vision_emb,dim=-1)
            
            # Append embeddings and labels to the respective lists
            text_embeddings.append(text_emb.cpu().numpy())
            audio_embeddings.append(audio_emb.cpu().numpy())
            vision_embeddings.append(vision_emb.cpu().numpy())
            labels.extend(label.cpu().numpy())

    # Convert the lists to numpy arrays
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    audio_embeddings = np.concatenate(audio_embeddings, axis=0)
    vision_embeddings = np.concatenate(vision_embeddings, axis=0)   

    #VISUALIZE
    visualize_3d(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels) 

    text_embeddings =   torch.from_numpy(text_embeddings) 
    audio_embeddings =  torch.from_numpy(audio_embeddings)
    vision_embeddings = torch.from_numpy(vision_embeddings)


    #EVAL 

    feat_t_new = []
    class_ids=[]
    input_ids_new = []
    attention_mask_new = []
    for i,text in enumerate(labels):

        if text not in class_ids:
            class_ids.append(text)
            feat_t_new.append(text_embeddings[i])


    text_embeddings = feat_t_new
    text_embeddings = torch.stack(text_embeddings,dim=0)

    volume = volume_computation3(text_embeddings, audio_embeddings, vision_embeddings)
    log = compute_metric_ret2(volume.T, class_ids, labels, direction='forward')
    log = {k.replace('forward','ZS CLASSIFICATION'): v for k,v in log.items()}
    print(log)
    wandb.log(log)






def visualize_3d(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):    
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
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

    # Labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Latent Space Visualization of Text, Audio, and Vision Embeddings')
    
    # Add legend
    ax.legend()
    plt.savefig(f'latent space at {iterations}.png')
    



# Updated train model with latent space visualization
def train_model_with_visualization(text_encoder, audio_encoder, vision_encoder, dataloader_train, dataloader_test, optimizer, device, num_iterations,contra_temp):
    text_encoder.train()
    audio_encoder.train()
    vision_encoder.train()
    iteration=0
    with torch.no_grad():
        eval(dataloader_test, text_encoder, audio_encoder, vision_encoder, device,iteration)
        text_encoder.train()
        audio_encoder.train()
        vision_encoder.train()

    running_loss = 0.0
        
    tq=tqdm(range(num_iterations),total=num_iterations)
    train_iterator = iter(dataloader_train)
    for batch_idx in tq:
        batch = next(train_iterator)
        mnist_img, spectogram, text_description, label = batch
        text_input = text_description
        audio_input = spectogram.to(device)
        vision_input = mnist_img.to(device)
        # Forward pass for all three modalities
        text_embedding = text_encoder(text_input).to(device)
        audio_embedding = audio_encoder(audio_input)
        vision_embedding = vision_encoder(vision_input)
        text_embedding = F.normalize(text_embedding,dim=-1)
        audio_embedding = F.normalize(audio_embedding,dim=-1)
        vision_embedding = F.normalize(vision_embedding,dim=-1)
        bs = text_embedding.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to('cuda')

        # LOSS COMPUTATION
        #VOLUME
        # volume = volume_computation3(text_embedding, audio_embedding, vision_embedding)
        # volume = volume / contra_temp
        # volumeT = volume_computation3(text_embedding, audio_embedding, vision_embedding).T
        # volumeT = volumeT / contra_temp

        # loss = (
        #         F.cross_entropy(-volume, targets, label_smoothing=0.1)
        #         + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        # ) / 2

        centroids = compute_centroids(text_embedding, audio_embedding, vision_embedding)
        centroids = centroids / contra_temp
        centroidsT = compute_centroids(text_embedding, audio_embedding, vision_embedding).T
        centroidsT = centroidsT / contra_temp
        lossCentr = (
                F.cross_entropy(centroids, targets, label_smoothing=0.1)
                + F.cross_entropy(centroidsT, targets, label_smoothing=0.1)
        ) / 2
        loss= lossCentr



        # #TV
        tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        tv = tv / contra_temp
        vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        vt = vt / contra_temp
        loss_tv = (
                F.cross_entropy(tv, targets, label_smoothing=0.1)
                + F.cross_entropy(vt, targets, label_smoothing=0.1)
        ) / 2
        #TA
        ta = torch.matmul(text_embedding, audio_embedding.permute(1,0))
        ta = ta / contra_temp
        at = torch.matmul(audio_embedding, text_embedding.permute(1,0))
        at = at / contra_temp
        loss_ta = (
                F.cross_entropy(ta, targets, label_smoothing=0.1)
                + F.cross_entropy(at, targets, label_smoothing=0.1)
        ) / 2
        
        
        loss = (loss + loss_tv + loss_ta)/3



        # if batch_idx<0:
        #     #VOLUME
        #     volume = volume_computation3(text_embedding, audio_embedding, vision_embedding)
        #     volume = volume / contra_temp
        #     volumeT = volume_computation3(text_embedding, audio_embedding, vision_embedding).T
        #     volumeT = volumeT / contra_temp

        #     loss = (
        #             F.cross_entropy(-volume, targets, label_smoothing=0.1)
        #             + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        #     ) / 2


        #     # CENTROIDS LOSS
        #     centroids = compute_centroids(text_embedding, audio_embedding, vision_embedding)
        #     centroids = centroids / contra_temp
        #     centroidsT = compute_centroids(text_embedding, audio_embedding, vision_embedding).T
        #     centroidsT = centroidsT / contra_temp

        #     lossCentr = (
        #             F.cross_entropy(centroids, targets, label_smoothing=0.1)
        #             + F.cross_entropy(centroidsT, targets, label_smoothing=0.1)
        #     ) / 2
            
        #     loss= (loss+lossCentr)/2


        # if batch_idx >= 0:
        # # #TV
        #     tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        #     tv = tv / contra_temp
        #     vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        #     vt = vt / contra_temp
        #     loss_tv = (
        #             F.cross_entropy(tv, targets, label_smoothing=0.1)
        #             + F.cross_entropy(vt, targets, label_smoothing=0.1)
        #     ) / 2

        #     #TA
        #     ta = torch.matmul(text_embedding, audio_embedding.permute(1,0))
        #     ta = ta / contra_temp
        #     at = torch.matmul(audio_embedding, text_embedding.permute(1,0))
        #     at = at / contra_temp
        #     loss_ta = (
        #             F.cross_entropy(ta, targets, label_smoothing=0.1)
        #             + F.cross_entropy(at, targets, label_smoothing=0.1)
        #     ) / 2

        #     loss = (loss_tv+loss_ta)/2


            # #AV
            # av = torch.matmul(audio_embedding, vision_embedding.permute(1,0))
            # av = av / contra_temp
            # va = torch.matmul(vision_embedding, audio_embedding.permute(1,0))
            # va = va / contra_temp
            # loss_va = (
            #         F.cross_entropy(av, targets, label_smoothing=0.1)
            #         + F.cross_entropy(va, targets, label_smoothing=0.1)
            # ) / 2
            # loss = (loss+loss_tv+loss_ta+loss_va)/4
        

            # Log training loss to W&B
        wandb.log({
        "train_loss": loss,
        })
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #tq.set_description("Loss = %d" % loss.item())
        tq.set_postfix(loss=loss.item())
        running_loss += loss.item()
        iteration += 1
        with torch.no_grad():
            if (iteration % 50)==0:
                eval(dataloader_test, text_encoder, audio_encoder, vision_encoder, device,iteration) 
                text_encoder.train()
                audio_encoder.train()
                vision_encoder.train()

    # Average loss for the epoch
    epoch_loss = running_loss / num_iterations
    print(f' Loss mean: {epoch_loss:.4f}')
    wandb.log({
        "mean_train_loss": epoch_loss,
    })
        


       

