import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleConsistencyLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CycleConsistencyLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_pairs, negative_pairs):
        # Ensure the target tensor is resized to match the output tensor's shape
        pos_loss = 0
        neg_loss = 0
        num_pos_pairs = len(positive_pairs)
        num_neg_pairs = len(negative_pairs)

        for (z_i, z_j) in positive_pairs:
            pos_loss += F.cosine_similarity(z_i, z_j).mean()
        
        for (z_k, z_l) in negative_pairs:
            neg_loss += F.cosine_similarity(z_k, z_l).mean()

        pos_loss /= num_pos_pairs
        neg_loss /= num_neg_pairs

        loss = pos_loss + F.relu(self.margin - neg_loss)
        return loss

if __name__ == "__main__":
    loss = CycleConsistencyLoss()
    print(loss)
    z_i = torch.randn(10, 128)
    z_j = torch.randn(10, 128)
    z_k = torch.randn(10, 128)
    z_l = torch.randn(10, 128)
    positive_pairs = [(z_i[i], z_j[i]) for i in range(10)]
    negative_pairs = [(z_k[i], z_l[i]) for i in range(10)]
    loss_value = loss(positive_pairs, negative_pairs)
    print(f'Loss value: {loss_value.item()}')
