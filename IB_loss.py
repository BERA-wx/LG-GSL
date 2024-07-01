import torch
import torch.nn as nn
import torch.nn.functional as F


class IBLoss(nn.Module):
    def __init__(self, temp):
        super(IBLoss, self).__init__()
        self.temp = temp

    def forward(self, ori_feats, latent_feats, labels, r_negative=0.1):
        # normalization
        ori_feats = F.normalize(ori_feats, p=2, dim=1)
        latent_feats = F.normalize(latent_feats, p=2, dim=1)
        labels = torch.tensor(labels).cuda()

        # scores
        logits = torch.exp(torch.mm(ori_feats, latent_feats.t()) / self.temp)
        neg_logits = torch.zeros_like(logits)
        for label in torch.unique(labels):
            # positive & negative samples
            pos_samples_all = torch.where(labels == label)[0]
            neg_samples_all = torch.where(labels != label)[0]
            # distances
            neg_distances = torch.cdist(ori_feats[pos_samples_all], ori_feats[neg_samples_all], p=2)
            # topK distances
            neg_topK_indices = torch.topk(neg_distances, int(r_negative * neg_samples_all.shape[0]), largest=True).indices
            # final indices
            neg_indices = neg_samples_all[neg_topK_indices]
            # logit
            neg_logits[pos_samples_all[:, None], neg_indices.cuda()] = logits[pos_samples_all[:, None], neg_indices]
            NEG = neg_logits.sum(dim=-1)

            # information bottleneck loss
            pos_aug = torch.diag(logits)
            ib_loss = (-torch.log(pos_aug / (pos_aug + NEG))).mean()

        return ib_loss
