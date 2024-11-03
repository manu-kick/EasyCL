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

def area_computation(language, video, audio):


    #print(f"norm language= {torch.sum(language ** 2, dim=1)}")
    
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
'''

def area_computation(language, video, audio):
    res = []
    for i in range(language.shape[0]):
        l=[]
    
        for j in range(language.shape[0]):
            u = language[i] - video[j]
            u_norm = torch.tensor(u@ u.T)

            v = language[i] - audio[j]
            v_norm = torch.tensor(v@ v.T)

            uv_dot = u@v.T
            area = torch.sqrt( (u_norm * v_norm) - (uv_dot * uv_dot)) / 2
            #print(area)
            l.append(area.item())
        
        
        res.append(l)
        
    print(res[0][0])
    res = np.array(res)
    res = torch.from_numpy(res).to("cuda")
    return res


'''

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
    


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    # G = torch.stack([
    #     torch.stack([torch.ones_like(ll), lv, la], dim=-1),  # First row of the Gram matrix
    #     torch.stack([torch.ones_like(lv),torch.ones_like(vv), va], dim=-1),  # Second row of the Gram matrix
    #     torch.stack([torch.ones_like(la), torch.ones_like(va), torch.ones_like(aa)], dim=-1)  # Third row of the Gram matrix
    # ], dim=-2)
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

def volume_computation3Test(language, video, audio):

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
    


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa], dim=-1)  # Third row of the Gram matrix
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


def compute_centroids_OLD(text_embeddings, visual_embeddings, audio_embeddings, norm_threshold=0.0):
    """
    Computes the centroid for each pair of samples between text embeddings and visual/audio embeddings,
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Then applies norm conditioning only on the main diagonal.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual embeddings.
    - audio_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio embeddings.
    - norm_threshold (float): The threshold to apply conditioning on the norms.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the norms conditioned on the diagonal.
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroids.
    """

    # Get batch sizes
    batch_size1 = text_embeddings.shape[0]   # For text embeddings
    batch_size2 = visual_embeddings.shape[0]  # For visual/audio embeddings

    # Compute centroids by averaging text and visual/audio embeddings
    # Expand the dimensions to allow pairwise computation
    text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]
    audio_expanded = audio_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]

    # Compute the centroid by averaging the embeddings
    centroids = (text_expanded + visual_expanded + audio_expanded) / 3.0
    
    # Compute norms along the last dimension (feature_dim)
    centroid_norms = torch.norm(centroids, dim=-1)  # Shape: [batch_size1, batch_size2]

    # Create a mask for the diagonal elements (i == j)
    diagonal_mask = torch.eye(batch_size1, batch_size2, device=centroid_norms.device)

    # Apply norm conditioning only to the diagonal elements
    # For the diagonal: apply the threshold condition
    # For non-diagonal elements: leave unchanged (or set to 0, or some other operation)
    conditioned_norms = torch.where(diagonal_mask.bool(), 
                                    centroid_norms - norm_threshold,
                                    #torch.where(centroid_norms >= norm_threshold, centroid_norms, torch.tensor(0.0, dtype=centroid_norms.dtype)), 
                                    centroid_norms)

    return conditioned_norms, centroids



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

    # Get batch sizes
    batch_size1 = text_embeddings.shape[0]   # For text embeddings
    batch_size2 = visual_embeddings.shape[0]  # For visual/audio embeddings

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
    """
    Computes the centroid for each pair of samples between text embeddings and visual/audio embeddings,
    by calculating the mean of the corresponding feature vectors across the two modalities.

    Then applies norm conditioning only on the main diagonal.

    Parameters:
    - text_embeddings (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing text embeddings.
    - visual_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing visual embeddings.
    - audio_embeddings (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio embeddings.
    - norm_threshold (float): The threshold to apply conditioning on the norms.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the norms conditioned on the diagonal.
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2, feature_dim) representing the centroids.
    """

    # Get batch sizes
    batch_size1 = text_embeddings.shape[0]   # For text embeddings
    batch_size2 = visual_embeddings.shape[0]  # For visual/audio embeddings

    # Compute centroids by averaging text and visual/audio embeddings
    # Expand the dimensions to allow pairwise computation
    #text_expanded = text_embeddings.unsqueeze(1)  # Shape: [batch_size1, 1, feature_dim]
    #visual_expanded = visual_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]
    #audio_expanded = audio_embeddings.unsqueeze(0)  # Shape: [1, batch_size2, feature_dim]

    # Compute the centroid by averaging the embeddings
    centroids = (text_embeddings + visual_embeddings + audio_embeddings) / 3.0
    
    # Compute norms along the last dimension (feature_dim)
    #centroid_norms = torch.norm(centroids, dim=-1)  # Shape: [batch_size1, batch_size2]

    # Create a mask for the diagonal elements (i == j)
    #diagonal_mask = torch.eye(batch_size1, batch_size2, device=centroid_norms.device)

    # Apply norm conditioning only to the diagonal elements
    # For the diagonal: apply the threshold condition
    # For non-diagonal elements: leave unchanged (or set to 0, or some other operation)
    #conditioned_norms = torch.where(diagonal_mask.bool(), 
    #                                centroid_norms - norm_threshold,
    #                                #torch.where(centroid_norms >= norm_threshold, centroid_norms, torch.tensor(0.0, dtype=centroid_norms.dtype)), 
    #                                centroid_norms)

    return  centroids


import numpy as np



# text_embeddings =   torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float()#torch.rand(10, 3)
# visual_embeddings = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float()#torch.rand(10, 3)
# audio_embeddings =  torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float()#torch.rand(10, 3)

# text_embeddings = F.normalize(text_embeddings)
# visual_embeddings = F.normalize(visual_embeddings)
# audio_embeddings = F.normalize(audio_embeddings)
  
# centroids = compute_centroids(text_embeddings, visual_embeddings, audio_embeddings)

#print(centroids)


# # Define the number of items
# n_items = 10

# # Initialize an empty similarity matrix
# similarity_matrix = np.zeros((n_items, n_items))

# # Fill the matrix with similarity values iteratively
# for i in range(n_items):
#     similarity_matrix[i, i] = 1.0  # Perfect similarity with itself
    
#     # For items further away from item i (distance 1 to n_items - 1)
#     for dist in range(1, n_items):
#         # The target item index
#         j = i + dist
#         if j < n_items:
#             # The similarity value decreases as distance increases
#             similarity_value = 1.0 - dist * 0.1
#             similarity_matrix[i, j] = similarity_value
#             similarity_matrix[j, i] = similarity_value  # Ensure symmetry

# # Print the similarity matrix
# print(similarity_matrix)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

def compute_loss_anchor(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):
    
    if cf.anchor_selection == 'text':
        #CENTROID-Video Alignment
        tv = torch.matmul(text_embedding, vision_embedding.permute(1,0))
        tv = tv / contra_temp
        vt = torch.matmul(vision_embedding, text_embedding.permute(1,0))
        vt = vt / contra_temp
        if cf.similarity_matrix == 'everywhere':
            loss_tv = (
                    F.cross_entropy(tv, similarity_matrix, label_smoothing=0.1)
                    + F.cross_entropy(vt, similarity_matrix, label_smoothing=0.1)
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
                    F.cross_entropy(ta, similarity_matrix, label_smoothing=0.1)
                    + F.cross_entropy(at, similarity_matrix, label_smoothing=0.1)
            ) /2
        else:
            loss_ta = (
                    F.cross_entropy(ta, targets, label_smoothing=0.1)
                    + F.cross_entropy(at, targets, label_smoothing=0.1)
            ) /2

        return (loss_tv + loss_ta) / 2
    
    else:
        print('Anchor selection not yet implemented')
        return None


def compute_loss_centroids(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):

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

    loss= (loss_ca+loss_ct+loss_cv+ cf.centroid_scale*loss_centr)/4

    return loss

def compute_loss_volume(text_embedding, audio_embedding, vision_embedding, bs, targets, cf, similarity_matrix,contra_temp):
    
    # #OLD LOSSES 
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
    
    # #OLD LOSSES 
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