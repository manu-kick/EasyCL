# equation 3/4 from "Semantic Compression via multimodal representation learning"
import torch 


def fisher_ratio(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    modalities_embeddings = dict()
    labels = torch.tensor(labels) #[bach_size]
    modalities_embeddings['text'] = torch.Tensor(text_embeddings) #[batch_size, embedding_dim]
    modalities_embeddings['audio'] = torch.Tensor(audio_embeddings) #[batch_size, embedding_dim]
    modalities_embeddings['vision'] = torch.Tensor(vision_embeddings) #[batch_size, embedding_dim]
    
    # get the number of samples per class
    num_classes = len(torch.unique(labels))
    N_c = torch.zeros(num_classes) # samples per class
    per_class_indices = dict()
    mu_c = dict() # mean per class (cluster centroid) per modality
    modality_mu = dict() # global mean per modality
    modality_mu['text'] = modalities_embeddings['text'].mean(dim=0) #[embedding_dim]
    modality_mu['audio'] = modalities_embeddings['audio'].mean(dim=0) #[embedding_dim]
    modality_mu['vision'] = modalities_embeddings['vision'].mean(dim=0) #[embedding_dim]
    
    for modality, embeddings in modalities_embeddings.items():
        mu_c[modality] = torch.zeros(num_classes, embeddings.shape[1]) # [num_classes, embedding_dim]
    
    # collect per class statistics
    for c in range(num_classes):
        N_c[c] = (labels == c).sum()
        per_class_indices[c] = (labels == c).nonzero(as_tuple=True)[0]
        for modality, embeddings in modalities_embeddings.items():
            mu_c[modality][c] = embeddings[per_class_indices[c]].mean(dim=0) # [embedding_dim]
    
    SB = dict() # [scalar]
    SW = dict() # [scalar]
    for c in range(num_classes):
        for modality, embeddings in modalities_embeddings.items():
            # Between-class scatter
            diff_mu = (mu_c[modality][c] - modality_mu[modality]).unsqueeze(1) # [embedding_dim, 1]
            SB_modality = N_c[c] * (diff_mu @ diff_mu.t()) # [embedding_dim, embedding_dim]
            if modality not in SB:
                SB[modality] = SB_modality
            else:
                SB[modality] += SB_modality
            
            # Within-class scatter
            SW_modality = torch.zeros(embeddings.shape[1], embeddings.shape[1]) # [embedding_dim, embedding_dim]
            for idx in per_class_indices[c]:
                diff_x = (embeddings[idx] - mu_c[modality][c]).unsqueeze(1) # [embedding_dim, 1]
                SW_modality += (diff_x @ diff_x.t()) # [embedding_dim, embedding_dim]
            if modality not in SW:
                SW[modality] = SW_modality
            else:
                SW[modality] += SW_modality
        
        
        
    
    
    pass

def cumulative_explained_variance(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    pass

def fisher_and_cumulative_explained_variance(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    return fisher_ratio(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels), cumulative_explained_variance(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)