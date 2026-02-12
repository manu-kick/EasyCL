# This files implements several computation of the modality gap as described in several papers

# Modality gap as in the original paper
import torch
import wandb


def L2M(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    # Compute the mean for each modality
    # Compute the L2 distance between the means taken pairwise
    # Also compute the overall average between the three pairs
    text_embeddings = torch.Tensor(text_embeddings)
    audio_embeddings = torch.Tensor(audio_embeddings)
    vision_embeddings = torch.Tensor(vision_embeddings)
    
    mean_text = text_embeddings.mean(dim=0)
    mean_audio = audio_embeddings.mean(dim=0)   
    mean_vision = vision_embeddings.mean(dim=0)
    
    l2_text_audio = torch.norm(mean_text - mean_audio, p=2).item()
    l2_text_vision = torch.norm(mean_text - mean_vision, p=2).item()
    l2_audio_vision = torch.norm(mean_audio - mean_vision, p=2).item()
    
    overall_l2 = (l2_text_audio + l2_text_vision + l2_audio_vision) / 3
    
    if cf.wandb:
        wandb.log({f'{metric}_gap': {
            'text_audio': l2_text_audio,
            'text_vision': l2_text_vision,
            'audio_vision': l2_audio_vision,
            'overall_l2': overall_l2
        }})
    
    return {
        'text_audio': l2_text_audio,
        'text_vision': l2_text_vision,
        'audio_vision': l2_audio_vision,
        'overall_l2': overall_l2
    }
    
# Relative modality gap as in https://openreview.net/pdf?id=uAFHCZRmXk
def rmg_numerator(mod1, mod2):
    # compute the average cosine dissimilarity (1-cos(x,y))/2 between matching pairs
    return torch.mean(1 - torch.nn.functional.cosine_similarity(mod1, mod2)).item()

def rmg_denominator(mod1, mod2 , numerator):
    N = mod1.shape[0]
    factor_multiplier = 1/((2*N)*(N-1))
    # ---- Intra-modality dissimilarities ----
    # Cosine similarity matrices
    sim_mod1 = torch.nn.functional.cosine_similarity(mod1.unsqueeze(1), mod1.unsqueeze(0), dim=-1)  # [N, N]
    sim_mod2 = torch.nn.functional.cosine_similarity(mod2.unsqueeze(1), mod2.unsqueeze(0), dim=-1)  # [N, N]
    
    # Cosine dissimilarity matrices (1 - similarity) / 2
    dissim_mod1 = (1 - sim_mod1) / 2  # [N, N]
    dissim_mod2 = (1 - sim_mod2) / 2  # [N, N]
    
    # We only want the upper triangle (excluding diagonal) for intra-modality pairs
    intra_mod1 = dissim_mod1.triu(diagonal=1).sum().item()
    intra_mod2 = dissim_mod2.triu(diagonal=1).sum().item()
    
    return (factor_multiplier * (intra_mod1 + intra_mod2))+ numerator
    
    
    

def RMG(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    couple_modalities = [('text','audio'),('text','vision'),('audio','vision')]
    
    embeddings = {
        'text':  torch.Tensor(text_embeddings),
        'audio': torch.Tensor(audio_embeddings),
        'vision': torch.Tensor(vision_embeddings)
    }
    rmg = dict()
    for couple in couple_modalities:
        mod1, mod2 = couple
        numerator = rmg_numerator(embeddings[mod1], embeddings[mod2])
        denominator = rmg_denominator(embeddings[mod1], embeddings[mod2], numerator)
        rmg_value = numerator / denominator
        rmg[f'{mod1}_{mod2}'] = rmg_value
        
        
        if cf.wandb:
            wandb.log({f'{metric}_gap': {
                f'{mod1}_{mod2}': rmg_value,
            }})
            
    overall_rmg = sum(rmg.values()) / len(rmg)
    rmg['overall'] = overall_rmg
    if cf.wandb:
        wandb.log({f'{metric}_gap': {
            'mean': overall_rmg
        }})
            
    


# L2 Instance gap
def L2I(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    # average l2 norm between matching pairs
    text_embeddings = torch.Tensor(text_embeddings)
    audio_embeddings = torch.Tensor(audio_embeddings)
    vision_embeddings = torch.Tensor(vision_embeddings)
    
    # count_samples = text_embeddings.shape[0]
    l2i_text_audio = torch.norm(text_embeddings - audio_embeddings, p=2, dim=-1).mean().item()
    l2i_text_vision = torch.norm(text_embeddings - vision_embeddings, p=2, dim=-1).mean().item()
    l2i_audio_vision = torch.norm(audio_embeddings - vision_embeddings, p=2, dim=-1).mean().item()
    
    overall_l2i = (l2i_text_audio + l2i_text_vision + l2i_audio_vision) / 3
    if cf.wandb:
        wandb.log({f'{metric}_gap': {
            'text_audio': l2i_text_audio,
            'text_vision': l2i_text_vision,
            'audio_vision': l2i_audio_vision,
            'overall_l2i': overall_l2i
        }})
        
    return {
        'text_audio': l2i_text_audio,  
        'text_vision': l2i_text_vision,
        'audio_vision': l2i_audio_vision,
        'overall_l2i': overall_l2i
    }

def compute_gap(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    if metric=='L2M':
        L2M(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    elif metric=='RMG':
        RMG(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    elif metric=='L2I':
        L2I(cf,metric,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels)
    else:
        raise ValueError(f'Unknown metric {metric}')
 