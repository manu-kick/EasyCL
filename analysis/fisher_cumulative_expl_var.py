# equation 3/4 from "Semantic Compression via multimodal representation learning"
import torch 
import wandb

def fisher_ratio(cf, text_embeddings, audio_embeddings, vision_embeddings, iterations, labels, eps=1e-12):
    modalities_embeddings = dict()
    labels = torch.as_tensor(labels).long()  # [batch_size]

    modalities_embeddings['text'] = torch.as_tensor(text_embeddings).float()    # [N, D]
    modalities_embeddings['audio'] = torch.as_tensor(audio_embeddings).float()  # [N, D]
    modalities_embeddings['vision'] = torch.as_tensor(vision_embeddings).float()# [N, D]

    D = modalities_embeddings['text'].shape[1]

    # Global mean across ALL modality samples (so 3N samples total)
    global_mean = torch.cat(
        [modalities_embeddings['vision'], modalities_embeddings['text'], modalities_embeddings['audio']],
        dim=0
    ).mean(dim=0)  # [D]

    classes = torch.unique(labels)
    num_classes = len(classes)

    mu_c = {}
    N_c = torch.zeros(num_classes, dtype=torch.long)

    # Compute per-class centroid (across modalities, consistent with global_mean)
    for k, c in enumerate(classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        N_c[k] = idx.numel()

        Xc = torch.cat([
            modalities_embeddings['text'][idx],
            modalities_embeddings['audio'][idx],
            modalities_embeddings['vision'][idx],
        ], dim=0)  # [3*Nc, D]

        mu_c[k] = Xc.mean(dim=0)  # [D]

    SB = torch.zeros((D, D), dtype=torch.float32)
    SW = torch.zeros((D, D), dtype=torch.float32)

    for k in range(num_classes):
        idx = (labels == classes[k]).nonzero(as_tuple=True)[0]

        # Between-class scatter (your original weighting kept)
        diff_mu = (mu_c[k] - global_mean).unsqueeze(1)  # [D, 1]
        SB += N_c[k].float() * (diff_mu @ diff_mu.t())  # [D, D]

        # Within-class scatter (FIXED): sum over all samples in class (and modalities)
        Xc = torch.cat([
            modalities_embeddings['text'][idx],
            modalities_embeddings['audio'][idx],
            modalities_embeddings['vision'][idx],
        ], dim=0)  # [3*Nc, D]

        centered = Xc - mu_c[k].unsqueeze(0)            # [3*Nc, D]
        SW += centered.t() @ centered                   # [D, D]

    TR_SB = torch.trace(SB)
    TR_SW = torch.trace(SW)

    fisher_score = TR_SB / (TR_SW + eps)
    if iterations % 100 == 0 and cf.wandb:
        wandb.log({f"fisher_score": fisher_score.item()})
        
    return fisher_score, SB, SW

def cumulative_explained_variance(cf, SB, SW, iterations,  eps: float = 1e-12):
    """
    Compute cumulative explained variance (CEV) from scatter matrices.

    We follow the standard PCA-style definition:
        - Build a "total scatter" matrix S_T = S_B + S_W
        - Compute its eigenvalues (variance along each principal direction)
        - Sort eigenvalues in descending order
        - CEV(k) = sum_{i=1..k} λ_i / sum_{i=1..D} λ_i

    Args:
        SB: [D, D] between-class scatter matrix
        SW: [D, D] within-class scatter matrix
        eps: small constant to avoid division by zero

    Returns:
        cev: [D] cumulative explained variance curve
        eigvals_sorted: [D] eigenvalues sorted descending
        ST: [D, D] total scatter matrix used for decomposition
    """
    # 1) Ensure we are working in floating point (needed for eigendecomposition).
    SB = SB.float()
    SW = SW.float()
    
    # 2) Total scatter: captures overall variability.
    #    In classical LDA/PCA relations, S_T = S_B + S_W.
    ST = SB + SW  # [D, D]
    
    
    # 3) Numerical sanity: make sure ST is symmetric.
    #    In theory it is symmetric, but tiny floating errors can appear.
    #    Symmetrizing improves stability for eigenvalue routines.
    ST = 0.5 * (ST + ST.t())
    
    
    # 4) Compute eigenvalues of a symmetric matrix.
    #    eigvalsh is specialized for Hermitian/symmetric matrices and returns real eigenvalues.
    eigvals = torch.linalg.eigvalsh(ST)  # [D], ascending order by default
    
    # 5) Clamp very small negative eigenvalues caused by numerical noise.
    #    Scatter matrices should be PSD, so negatives are usually just floating error.
    eigvals = torch.clamp(eigvals, min=0.0)

    # 6) Reverse to descending order so "component 1" is the direction of largest variance.
    eigvals_sorted = torch.flip(eigvals, dims=[0])  # [D], descending

    # 7) Compute the cumulative sum of eigenvalues: numerator for each k.
    cumsum = torch.cumsum(eigvals_sorted, dim=0)  # [D]

    # 8) Total variance is the sum of all eigenvalues (same as trace(ST)).
    total = eigvals_sorted.sum()  # scalar

    # 9) Normalize cumulative sums to get CEV(k) in [0, 1].
    #    eps avoids division-by-zero in degenerate cases.
    cev = cumsum / (total + eps)  # [D]

    # 10) Return the curve plus extras useful for plotting/inspection.
    
    if iterations % 100 == 0 and cf.wandb:
        wandb.log({f"cumulative_explained_variance": cev.numpy()})
        
    return cev, eigvals_sorted, ST

def fisher_and_cumulative_explained_variance(cf,text_embeddings,audio_embeddings,vision_embeddings,iterations,labels):
    fisher_score, SB, SW = fisher_ratio(cf, text_embeddings, audio_embeddings, vision_embeddings, iterations, labels)
    cev, eigvals_sorted, ST = cumulative_explained_variance(cf, SB, SW, iterations=iterations)
    return fisher_score, cev, eigvals_sorted