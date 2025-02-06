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
import numpy as np

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()


def wasserstein_distance(x, y):
    """
    Compute the Wasserstein distance (Earth Mover's Distance) between two 1D distributions.
    
    Args:
    - x: A tensor of shape (N,) representing the first distribution (real data).
    - y: A tensor of shape (M,) representing the second distribution (generated data).
    
    Returns:
    - wasserstein_distance: A scalar tensor representing the Wasserstein distance.
    """
    
    # Sort both sets of points
    x_sorted = torch.sort(x).values
    y_sorted = torch.sort(y).values
    
    # Compute the sum of absolute differences (simplified Wasserstein distance)
    distance = torch.abs(x_sorted - y_sorted).sum()
    
    return distance

def area_computation(language, video, audio):

    
    language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

    # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
    u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
    v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

    # Compute the norms for u and v
    u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
    v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

    # Compute the dot products for all pairs
    uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

    # Calculate the area for all pairs. I remove sqrt calculation
    area = ((u_norm * v_norm) - (uv_dot ** 2))/2#torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
    
    return area


def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    

    G = torch.stack([
        torch.stack([torch.ones_like(ll), lv, la], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, torch.ones_like(vv), va], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, torch.ones_like(aa)], dim=-1)  # Third row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation5(language, video, audio, subtitles, depth):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T
    ld = language@depth.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    dd = torch.einsum('bi,bi->b', depth, depth).unsqueeze(0).expand(batch_size1, -1)
    dv = torch.einsum('bi,bi->b', depth, video).unsqueeze(0).expand(batch_size1, -1)
    da = torch.einsum('bi,bi->b', depth, audio).unsqueeze(0).expand(batch_size1, -1) 
    ds = torch.einsum('bi,bi->b', depth, subtitles).unsqueeze(0).expand(batch_size1, -1)


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls, ld], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs, dv], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa, da], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss, ds], dim=-1),   # Fourth row of the Gram matrix
        torch.stack([ld, dv, da, ds, dd], dim=-1)
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res




def compute_centroidsTest(text_embeddings, visual_embeddings, audio_embeddings):
    """
    Computes the centroid for each pair of samples between text embeddings and visual/audio embeddings
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_audio_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual/audio embeddings.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroid for each pair.
    """

    # Compute centroids by averaging text and visual/audio embeddings
    # Expand the dimensions to allow pairwise computation
    text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]
    audio_expanded = audio_embeddings.unsqueeze(0)

    # Compute the centroid by averaging the embeddings
    
    centroids = (text_expanded + visual_expanded + audio_expanded) / 3.0
    #print(centroids)

    
    centroid_norms = torch.norm(centroids, dim=-1)

    #norm_condition = centroid_norms >= 0.5
    #centroid_norms = torch.where(norm_condition, centroid_norms, torch.tensor(0.0))

    return centroid_norms, centroids


def compute_centroids_only(text_embeddings, visual_embeddings, audio_embeddings, norm_threshold=0.0):

    # Compute centroids by averaging text and visual/audio embeddings
    centroids = (text_embeddings + visual_embeddings + audio_embeddings) / 3.0

    return  centroids


def compute_loss_anchor(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):
    
    #contra_temp = 0.1
    if cf.anchor_selection == 'text':
        #CENTROID-Video Alignment
        tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        tv = tv / contra_temp
        vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        vt = vt / contra_temp
        if cf.similarity_matrix == 'everywhere':
            loss_tv = (
                    F.cross_entropy(tv, similarity_matrix)#, label_smoothing=0.1)
                    + F.cross_entropy(vt, similarity_matrix)#, label_smoothing=0.1)
            ) / 2
        else:
            loss_tv = (
                    F.cross_entropy(tv, targets)#, label_smoothing=0.1)
                    + F.cross_entropy(vt, targets)#, label_smoothing=0.1)
            ) / 2

        #CENTROID-Audio Alignment
        ta = torch.matmul(text_embedding, audio_embedding.permute(1,0))
        ta = ta / contra_temp
        at = torch.matmul(audio_embedding, text_embedding.permute(1,0))
        at = at / contra_temp

        if cf.similarity_matrix == 'everywhere':
            loss_ta = (
                    F.cross_entropy(ta, similarity_matrix)#, label_smoothing=0.1)
                    + F.cross_entropy(at, similarity_matrix)#, label_smoothing=0.1)
            ) /2
        else:
            loss_ta = (
                    F.cross_entropy(ta, targets)#, label_smoothing=0.1)
                    + F.cross_entropy(at, targets)#, label_smoothing=0.1)
            ) /2
        
        return (loss_tv + loss_ta) / 2
    
    else:
        print('Anchor selection not yet implemented')
        return None
    
def compute_loss_anchor_lunif_lalign(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):
    
    #contra_temp = 0.05
    if cf.anchor_selection == 'text':
        #CENTROID-Video Alignment
        tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        tv = tv / contra_temp
        vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        vt = vt / contra_temp
        if cf.similarity_matrix == 'everywhere':
            loss_tv = (
                    F.cross_entropy(tv, similarity_matrix)#, label_smoothing=0.1)
                    + F.cross_entropy(vt, similarity_matrix)#, label_smoothing=0.1)
            ) / 2
        else:
            loss_tv = (
                    F.cross_entropy(tv, targets, label_smoothing=0.1)
                    + F.cross_entropy(vt, targets, label_smoothing=0.1)
            ) / 2

        #CENTROID-Audio Alignment
        ta = torch.matmul(text_embedding, audio_embedding.permute(1,0))
        ta = ta / contra_temp
        at = torch.matmul(audio_embedding, text_embedding.permute(1,0))
        at = at / contra_temp

        if cf.similarity_matrix == 'everywhere':
            loss_ta = (
                    F.cross_entropy(ta, similarity_matrix)#, label_smoothing=0.1)
                    + F.cross_entropy(at, similarity_matrix)#, label_smoothing=0.1)
            ) /2
        else:
            loss_ta = (
                    F.cross_entropy(ta, targets, label_smoothing=0.1)
                    + F.cross_entropy(at, targets, label_smoothing=0.1)
            ) /2

        
        centroids = compute_centroids_only(text_embedding, audio_embedding, vision_embedding)

        centroids_normalized = F.normalize(centroids,dim=-1)
        

        loss_unif = 1.0* lunif(centroids_normalized)

    

        tv = lalign(text_embedding, vision_embedding)
        ta = lalign(text_embedding, audio_embedding)

        
        return (loss_tv + loss_ta)/2 + loss_unif + ta + tv 
    
    else:
        print('Anchor selection not yet implemented')
        return None


def compute_loss_centroids(text_embedding, audio_embedding, vision_embedding, iterations, targets, cf, similarity_matrix,contra_temp):

    centroids = compute_centroids_only(text_embedding, audio_embedding, vision_embedding)

    if cf.normalization_centroids:
        centroids_normalized = F.normalize(centroids,dim=-1)
        centroids_matrix = torch.matmul(centroids_normalized, centroids_normalized.permute(1,0))

        if cf.diag_centroid_normalized:
            centroids_norm =  torch.norm(centroids, dim=-1)
            mask = torch.diag(torch.ones_like(centroids_matrix))
            centroids_matrix = mask*torch.diag(centroids_norm) + (1. - mask)*centroids_matrix


    else:
        centroids_matrix = torch.matmul(centroids, centroids.permute(1,0))
    
    if cf.centroids_matrix_temperature:
        centroids_matrix = centroids_matrix / contra_temp
        #print(centroids_matrix)

    if cf.distribution_type == 'ce':   
        if cf.label_smoothing_centroids:
            if cf.similarity_matrix == 'false':
                loss_centr = F.cross_entropy(centroids_matrix, targets, label_smoothing=0.1)
            else:
                loss_centr = F.cross_entropy(centroids_matrix, similarity_matrix, label_smoothing=0.1)
        else:
            if cf.similarity_matrix == 'false':
                loss_centr = F.cross_entropy(centroids_matrix, targets)
            else:
                loss_centr = F.cross_entropy(centroids_matrix, similarity_matrix)
    elif cf.distribution_type == 'kl':

        centroids_matrix = F.log_softmax(centroids_matrix, dim=0)
        loss_centr = F.kl_div(centroids_matrix,similarity_matrix,reduction='batchmean')
    
    elif cf.distribution_type == 'wd':
        centroids_matrix = F.softmax(centroids_matrix, dim=0)
        loss_centr = wasserstein_distance(centroids_matrix, similarity_matrix)
        
    #CENTROID-Video Alignment
    if cf.detach_centroids:
        cv = torch.matmul(centroids.detach(), vision_embedding.permute(1,0))
        cv = cv / contra_temp
        vc = torch.matmul(vision_embedding, centroids.permute(1,0).detach())
        vc = vc / contra_temp
    else:
        cv = torch.matmul(centroids, vision_embedding.permute(1,0))
        cv = cv / contra_temp
        vc = torch.matmul(vision_embedding, centroids.permute(1,0))
        vc = vc / contra_temp
        
    if cf.similarity_matrix == 'everywhere':
        loss_cv = (
                F.cross_entropy(cv, similarity_matrix, label_smoothing=0.1)
                + F.cross_entropy(vc, similarity_matrix, label_smoothing=0.1)
        ) / 2
    else:
        loss_cv = (
                F.cross_entropy(cv, targets, label_smoothing=0.1)
                + F.cross_entropy(vc, targets, label_smoothing=0.1)
        ) / 2

    #CENTROID-Audio Alignment
    if cf.detach_centroids:
        ca = torch.matmul(centroids.detach(), audio_embedding.permute(1,0))
        ca = ca / contra_temp
        ac = torch.matmul(audio_embedding, centroids.permute(1,0).detach())
        ac = ac / contra_temp
    else:
        ca = torch.matmul(centroids, audio_embedding.permute(1,0))
        ca = ca / contra_temp
        ac = torch.matmul(audio_embedding, centroids.permute(1,0))
        ac = ac / contra_temp

    if cf.similarity_matrix == 'everywhere':
        loss_ca = (
                F.cross_entropy(ca, similarity_matrix, label_smoothing=0.1)
                + F.cross_entropy(ac, similarity_matrix, label_smoothing=0.1)
        ) /2
    else:
        loss_ca = (
                F.cross_entropy(ca, targets, label_smoothing=0.1)
                + F.cross_entropy(ac, targets, label_smoothing=0.1)
        ) /2


    #Centroid-Text Alignment
    if cf.detach_centroids:
        ct = torch.matmul(centroids.detach(), text_embedding.permute(1,0))
        ct = ct / contra_temp
        tc = torch.matmul(text_embedding, centroids.permute(1,0).detach())
        tc = tc / contra_temp
    else:
        ct = torch.matmul(centroids, text_embedding.permute(1,0))
        ct = ct / contra_temp
        tc = torch.matmul(text_embedding, centroids.permute(1,0))
        tc = tc / contra_temp
        
    if cf.similarity_matrix == 'everywhere':
        loss_ct = (
                F.cross_entropy(ct, similarity_matrix, label_smoothing=0.1)
                + F.cross_entropy(tc, similarity_matrix, label_smoothing=0.1)
        ) / 2
    else:
        loss_ct = (
                F.cross_entropy(ct, targets, label_smoothing=0.1)
                + F.cross_entropy(tc, targets, label_smoothing=0.1)
        ) / 2

    loss= (loss_ca+loss_ct+loss_cv + cf.centroid_scale*loss_centr)/4

    return loss

def compute_loss_volume(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):
    
    # #VOLUME
    volume = volume_computation3(text_embedding, audio_embedding, vision_embedding)
    volume = volume / contra_temp
    volumeT = volume_computation3(text_embedding, audio_embedding, vision_embedding).T
    volumeT = volumeT / contra_temp

    if cf.similarity_matrix == 'everywhere':
        loss_vol = (
                F.cross_entropy(-volume, similarity_matrix, label_smoothing=0.1)
                + F.cross_entropy(-volumeT, similarity_matrix, label_smoothing=0.1)
        ) / 2
    else:
        loss_vol = (
            F.cross_entropy(-volume, targets, label_smoothing=0.1)
            + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        ) / 2
    
    return loss_vol


def compute_loss_area(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):

    # #VOLUME
    volume = area_computation(text_embedding, audio_embedding, vision_embedding)
    volume = volume / contra_temp
    volumeT = area_computation(text_embedding, audio_embedding, vision_embedding).T
    volumeT = volumeT / contra_temp

    if cf.similarity_matrix == 'everywhere':
        loss_vol = (
                F.cross_entropy(-volume, similarity_matrix, label_smoothing=0.1)
                + F.cross_entropy(-volumeT, similarity_matrix, label_smoothing=0.1)
        ) / 2
    else:
        loss_vol = (
            F.cross_entropy(-volume, targets, label_smoothing=0.1)
            + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        ) / 2
    
    return loss_vol