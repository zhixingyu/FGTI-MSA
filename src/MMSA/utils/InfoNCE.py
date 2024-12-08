import torch
import torch.nn.functional as F

def contralLoss(representations, label, temperature=0.1):
    T = temperature
    label = label
    n = label.shape[0]
    representations = representations
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
    mask_no_sim = torch.ones_like(mask) - mask
    mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )
    similarity_matrix = torch.exp(similarity_matrix/T)
    similarity_matrix = similarity_matrix*mask_dui_jiao_0.to(device='cuda')
    sim = mask*similarity_matrix
    no_sim = similarity_matrix - sim
    no_sim_sum = torch.sum(no_sim , dim=1)
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum  = sim + no_sim_sum_expend
    loss = torch.div(sim , sim_sum)
    loss = mask_no_sim.to(device='cuda') + loss.to(device='cuda') + torch.eye(n, n ).to(device='cuda')
    loss = -torch.log(loss)
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
    return loss
