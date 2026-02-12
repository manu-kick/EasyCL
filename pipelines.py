import torch
import torch.nn.functional as F
import numpy as np
from loss import volume_computation3,compute_centroidsTest
from loss import compute_loss_anchor,compute_loss_centroids,compute_loss_volume,area_computation,compute_loss_area,compute_loss_anchor_lunif_lalign
from tqdm import tqdm
import wandb
from metrics import compute_metric_ret,compute_metric_ret2
from utils import visualize_3d,compute_similarity_matrix,visualize_3d_interactively
from analysis.gap_mean_differences import  gap_mean_differences
from analysis.gap_embedding_dim_pairs import gap_embedding_dim_pairs
from analysis.fisher_cumulative_expl_var import fisher_and_cumulative_explained_variance
from analysis.intrinsic_dimensions import intrinsic_dimension_mle
from analysis.modality_gap import compute_gap


def eval(cf, test_dataloader, text_encoder, audio_encoder, vision_encoder, device,iterations):
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

            # Normalize embeddings
            text_emb = F.normalize(text_emb,dim=-1)
            audio_emb = F.normalize(audio_emb,dim=-1)
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

    # ---------- RUN ANALYSIS AND VISUALIZATION ----------
    visualize_3d(cf, text_embeddings,audio_embeddings,vision_embeddings,iterations,labels) 
    gap_mean_differences(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels) 
    gap_embedding_dim_pairs(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    fisher_and_cumulative_explained_variance(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    intrinsic_dimension_mle(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    for i in ['L2M','RMG','L2I']:
        compute_gap(cf,i,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    
    if iterations>5000 and iterations<5051 :
        visualize_3d_interactively(text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)

    if iterations>8000 and iterations<8051 :
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

    if cf.eval_type == 'volume':
        volume = volume_computation3(text_embeddings, audio_embeddings, vision_embeddings)
        log = compute_metric_ret2(volume.T, class_ids, labels, direction='forward')
    elif cf.eval_type == 'centroids':
        centroids_norm,centroids = compute_centroidsTest(text_embeddings, audio_embeddings, vision_embeddings)
        log = compute_metric_ret(centroids_norm.T, class_ids, labels, direction='forward')
    elif cf.eval_type == 'area':
        area = area_computation(text_embeddings, audio_embeddings, vision_embeddings)
        log = compute_metric_ret2(area.T, class_ids, labels, direction='forward')

    elif cf.eval_type == 'tvta':
        sim_tv = text_embeddings @ vision_embeddings.T
        sim_ta = text_embeddings @ audio_embeddings.T

        log1 = compute_metric_ret2(sim_tv, class_ids, labels, direction='forward')
        log2 = compute_metric_ret2(sim_ta, class_ids, labels, direction='forward')
        log = {}
        log['forward_r1'] = (log1['forward_r1']+log2['forward_r1'])/2
        log['forward_ravg'] = (log1['forward_ravg']+log2['forward_ravg'])/2

    elif cf.eval_type == 'symile':
        query = vision_embeddings * audio_embeddings
        sim = text_embeddings @ query.T
        log = compute_metric_ret2(sim, class_ids, labels, direction='forward')

    log = {k.replace('forward','ZS CLASSIFICATION'): v for k,v in log.items()}
    print(log)
    if cf.wandb:
        wandb.log(log)

def get_loss(loss_type, text_embedding, audio_embedding, vision_embedding, batch_idx, bs, targets, cf, similarity_matrix,contra_temp):
    if loss_type ==  'anchor':                     #anchor or volume or centroids
        loss = compute_loss_anchor(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)

    elif loss_type ==  'centroids':  
        loss = compute_loss_centroids(text_embedding, audio_embedding, vision_embedding, batch_idx, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'volume':
        loss = compute_loss_volume(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'area':
        loss = compute_loss_area(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)

    elif loss_type == 'anchor_align_unif':
        loss = compute_loss_anchor_lunif_lalign(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp)
    else:    
        print("loss not implemented")
        return 0
    
    return loss
    
    
# Updated train model with latent space visualization
def train_model_with_visualization(cf,text_encoder, audio_encoder, vision_encoder, dataloader_train, dataloader_test, optimizer, device, num_iterations,contra_temp):
    if not cf.contra_temp_learnable:
        contra_temp = cf.contra_temp_init

    similarity_matrix=None
    if not cf.similarity_matrix == 'false':
        similarity_matrix = compute_similarity_matrix(cf.similarity_type)
    if cf.softmax_similarity:
        similarity_matrix = similarity_matrix / cf.similarity_temperature
        similarity_matrix = F.softmax(similarity_matrix,dim=-1)

    text_encoder.train()
    audio_encoder.train()
    vision_encoder.train()
    iteration=0
    with torch.no_grad():
        eval(cf, dataloader_test, text_encoder, audio_encoder, vision_encoder, device,iteration)
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

        if cf.normalization:
            text_embedding = F.normalize(text_embedding,dim=-1)
            audio_embedding = F.normalize(audio_embedding,dim=-1)
            vision_embedding = F.normalize(vision_embedding,dim=-1)


        text_embedding = text_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices
        audio_embedding = audio_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices
        vision_embedding = vision_embedding[np.argsort(label)]  # Sort embeddings according to sorted indices

        bs = text_embedding.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to('cuda')
        loss = get_loss(cf.loss_type, text_embedding, audio_embedding, vision_embedding, batch_idx, bs, targets, cf, similarity_matrix,contra_temp)
        
        # Log training loss to W&B
        if cf.wandb:
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
                eval(cf, dataloader_test, text_encoder, audio_encoder, vision_encoder, device,iteration) 
                
                #if the temperature is learnable plot on wandb the current value
                if cf.contra_temp_learnable and cf.wandb:
                    wandb.log({"contra_temp": contra_temp.item()})
                    
                text_encoder.train()
                audio_encoder.train()
                vision_encoder.train()

    # Average loss for the epoch
    epoch_loss = running_loss / num_iterations
    print(f' Loss mean: {epoch_loss:.4f}')
    if cf.wandb:
        wandb.log({
            "mean_train_loss": epoch_loss,
        })

    return 0 
        


       

