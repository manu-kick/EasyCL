import torch

def compute_metric_ret(score_matrix, ids, ids_txt, direction='forward'):


    print(len(ids_txt),len(ids))
    print(score_matrix.shape)
    assert score_matrix.shape == (len(ids_txt),len(ids))

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }
    

    return eval_log

def compute_metric_ret2(score_matrix, ids, ids_txt, direction='forward'):
                                #ids_caption, class_ids

    print(len(ids_txt),len(ids))
    #print(score_matrix.shape)
    assert score_matrix.shape == (len(ids_txt),len(ids))
    #print(ids)
    #print(ids_txt)
    #print(score_matrix)

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=False)[1].tolist()
        #print(indice_matrix)
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            #print(gt_indice)
            rank.append(indice_matrix[i].index(gt_indice))
            #print(rank)
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }
    

    return eval_log


#This function computes the gap between two modalities.
#The gap is computed as the euclidean distance between the centroids of the two modalities
def compute_gap(feat_modality1, feat_modality2):
    #FEATURES SHOULD BE NORMALIZED

    modality1_centroid = torch.mean(feat_modality1,0)
    modality2_centroid = torch.mean(feat_modality2,0)

    gap = modality1_centroid - modality2_centroid
    norm_gap = torch.norm(gap)

    return norm_gap

def compute_mean_angular_value_of_a_modality(feat_modality):
    #norm_modality = torch.norm(feat_modality,dim=1)
    #norm_modality = norm_modality.unsqueeze(1)
    #feat_modality = feat_modality/norm_modality

    #FEATURES SHOULD BE NORMALIZED
    

    cos_sim = feat_modality@feat_modality.T
    #EXCLUDE THE DIAGONAL THAT IS 1 and make a matrix n-1xn-1
    # Initialize an empty list to hold the result
    result = []
    n=cos_sim.size(0)
    # Loop through the rows and columns, excluding the diagonal
    for i in range(n):
        row = []
        for j in range(n):
            if i != j:  # Exclude diagonal element
                row.append(cos_sim[i, j])
        result.append(row)
    # Convert the result list to a tensor
    cos_sim_no_diag = torch.tensor(result)

    mean = cos_sim_no_diag.mean()
    return mean

import torch

def uniformity(features_modality1, features_modality2):
    """
    Calculate the uniformity metric for two modalities based on their features.

    Args:
        features_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [bs, d].
        features_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [bs, d].

    Returns:
        float: Uniformity metric (-W2).
    """
    # Concatenate the features of the two modalities
    Z = torch.cat([features_modality1, features_modality2], dim=0)  # Shape: [2 * bs, d]

    # Compute the sample mean \mu_hat and covariance \Sigma
    mu_hat = torch.mean(Z, dim=0)  # Shape: [d]
    Sigma = torch.cov(Z.T)  # Shape: [d, d]

    # Calculate the trace and square root of the covariance matrix
    trace_Sigma = torch.trace(Sigma)  # Scalar
    sqrt_Sigma = torch.linalg.matrix_power(Sigma, 1 // 2)  # Matrix square root of Sigma, shape: [d, d]
    trace_sqrt_Sigma = torch.trace(sqrt_Sigma)  # Scalar

    # Dimensionality of the features
    m = Z.shape[1]

    # Compute the quadratic Wasserstein distance W2
    W2 = torch.sqrt(
        torch.norm(mu_hat)**2 + 1 + trace_Sigma - (2 / torch.sqrt(torch.tensor(m, dtype=Sigma.dtype))) * trace_sqrt_Sigma
    )

    # Return the uniformity metric (-W2)
    return -W2.item()


def mean_distance_of_true_pairs(features_modality1, features_modality2):
    #It is needed that the features are normalized
    #It is needed that the i-th feature of modality1 corresponds to the same textual description of i-th feature of modality2

    cosine_sim = torch.matmul(features_modality1, features_modality2.permute(1,0))

    cosine_sim_diag = torch.diag(cosine_sim)
    cosine_tv_mean = torch.mean(cosine_sim_diag)
    return cosine_tv_mean


# Example usage

# Generate some random data for testing
#bs, d = 10, 5
#features_modality1 = torch.randn(bs, d)
#features_modality2 = torch.randn(bs, d)
#
## Compute uniformity metric
#uniformity_metric = uniformity(features_modality1, features_modality2)
#print(f"Uniformity Metric: {uniformity_metric}")