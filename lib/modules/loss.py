import torch


class Criterion(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(Criterion, self).__init__()
        self.loss = torch.nn.BCELoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        losses = []
        for pred, target in zip(preds, targets):
            pred = pred.reshape(pred.size(0), -1)
            target = target.reshape(target.size(0), -1)
            n_pos = (target > 0).sum(dim=-1)
            be_loss = self.loss(pred, target)
            pt = torch.exp(-be_loss)
            loss = self.alpha * ((1-pt)**self.gamma) * be_loss
            losses.append((loss.sum(dim=-1) / (n_pos + 1e-6)).mean())

        return losses