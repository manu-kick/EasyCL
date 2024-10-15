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
from utils import visualize_3d,visualize_3d_interactively,compute_similarity_matrix





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
    if iterations>5000 and iterations<5051 :
        visualize_3d_interactively(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)

    if iterations>7999 and iterations<8051 :
        visualize_3d_interactively(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)

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

    #volume = volume_computation3(text_embeddings, audio_embeddings, vision_embeddings)
    #log = compute_metric_ret2(volume.T, class_ids, labels, direction='forward')

    centroids_norm,centroids = compute_centroidsTest(text_embeddings, audio_embeddings, vision_embeddings)
    log = compute_metric_ret(centroids_norm.T, class_ids, labels, direction='forward')
    log = {k.replace('forward','ZS CLASSIFICATION'): v for k,v in log.items()}
    print(log)
    wandb.log(log)


# Updated train model with latent space visualization
def train_model_with_visualization(text_encoder, audio_encoder, vision_encoder, dataloader_train, dataloader_test, optimizer, device, num_iterations,contra_temp):
    
    similarity_matrix = compute_similarity_matrix()
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


        text_embedding = text_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices
        audio_embedding = audio_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices
        vision_embedding = vision_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices

        bs = text_embedding.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to('cuda')

        centroids = compute_centroids_only(text_embedding, audio_embedding, vision_embedding)
        centroids_matrix = torch.matmul(centroids, centroids.permute(1,0))

        #THIS LINE FOR SEMANTIC LEARNING
        #MARGIN ON THE INFONCELOSS
        #COMMENT IF YOU WANT TO SKIP
        centroids_matrix = centroids_matrix+similarity_matrix

        #TEMPERATURE SCALING
        centroids_matrix = centroids_matrix / contra_temp
        
        #CROSS ENTROPY
        loss_centr = (
                F.cross_entropy(centroids_matrix, targets, label_smoothing=0.1)
                + F.cross_entropy(centroids_matrix.T, targets, label_smoothing=0.1)
        ) / 2

        #CENTROID-VIDEO Alignment
        cv = torch.matmul(centroids, vision_embedding.permute(1,0))
        #PROVA
        cv = cv + similarity_matrix
        cv = cv / contra_temp
        vc = torch.matmul(vision_embedding, centroids.permute(1,0))
        #PROVA
        vc = vc + similarity_matrix
        vc = vc / contra_temp
        loss_cv = (
                F.cross_entropy(cv, targets, label_smoothing=0.1)
                + F.cross_entropy(vc, targets, label_smoothing=0.1)
        ) / 2
        #CENTROID-Audio Alignment
        ca = torch.matmul(centroids, audio_embedding.permute(1,0))
        ca = ca + similarity_matrix
        ca = ca / contra_temp
        ac = torch.matmul(audio_embedding, centroids.permute(1,0))
        ac = ac + similarity_matrix
        ac = ac / contra_temp
        loss_ca = (
                F.cross_entropy(ca, targets, label_smoothing=0.1)
                + F.cross_entropy(ac, targets, label_smoothing=0.1)
        ) / 2
        #Centroid-Text Alignment
        ct = torch.matmul(centroids, text_embedding.permute(1,0))
        ct = ct + similarity_matrix
        ct = ct / contra_temp
        tc = torch.matmul(text_embedding, centroids.permute(1,0))
        tc = tc + similarity_matrix
        tc = tc / contra_temp
        loss_ct = (
                F.cross_entropy(ct, targets, label_smoothing=0.1)
                + F.cross_entropy(tc, targets, label_smoothing=0.1)
        ) / 2

        loss= (loss_ca+loss_ct+loss_cv+loss_centr)/4


        
        # #OLD LOSSES 
        # #VOLUME
        # volume = volume_computation3(text_embedding, audio_embedding, vision_embedding)
        # volume = volume / contra_temp
        # volumeT = volume_computation3(text_embedding, audio_embedding, vision_embedding).T
        # volumeT = volumeT / contra_temp

        # loss_vol = (
        #         F.cross_entropy(-volume, targets, label_smoothing=0.1)
        #         + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        # ) / 2

        # #TEXT VIDEO ALIGNMENT
        # #TV
        # tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        # tv = tv / contra_temp
        # vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        # vt = vt / contra_temp
        # loss_tv = (
        #         F.cross_entropy(tv, targets, label_smoothing=0.1)
        #         + F.cross_entropy(vt, targets, label_smoothing=0.1)
        # ) / 2
        # #TEXT AUDIO ALIGNMENT
        # ta = torch.matmul(text_embedding, audio_embedding.permute(1,0))
        # ta = ta / contra_temp
        # at = torch.matmul(audio_embedding, text_embedding.permute(1,0))
        # at = at / contra_temp
        # loss_ta = (
        #         F.cross_entropy(ta, targets, label_smoothing=0.1)
        #         + F.cross_entropy(at, targets, label_smoothing=0.1)
        # ) / 2
        
        
        #loss = (lossCentr + loss_tv + loss_ta)/3
        #loss = (lossCentr + loss_tv + loss_ta + mse)/4
        #loss = (lossCentr  + mse)/2

        # Log training loss to W&B
        wandb.log({
        "train_loss": loss,
        })

        #BACKWARD LOSS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #LOG ON TQDM BAR
        tq.set_postfix(loss=loss.item())
        running_loss += loss.item()
        iteration += 1
        #EVALUATE EVERY evaluate_iteration=50
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
        


       

