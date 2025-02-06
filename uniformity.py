
import numpy as np
import math 
import torch


def calc_wasserstein_distance(x):
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x-x_center).t(), x-x_center)/N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps
    mS = np.sqrt(np.diag(S))
    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 



# Example usage

bs, d = 64, 512
features_modality1 = torch.randn(bs, d)
features_modality2 = torch.randn(bs, d)
Z = torch.cat([features_modality1, features_modality2], dim=0) 

calc_wasserstein_distance(Z)