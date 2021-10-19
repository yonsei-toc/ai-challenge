import torch
import torch.nn as nn
import torchmetrics


class SequenceClassifier(nn.Module):
    def __init__(self, hidden_size, n_labels, p_drop):
        super(SequenceClassifier, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.proj = nn.Linear(hidden_size, n_labels)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=n_labels)

    def forward(self, features, labels):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)

        loss = self.loss(x, labels)
        accuracy = self.accuracy(x, labels)

        return x, loss, accuracy


class SequenceTagging(nn.Module):
    def __init__(self, hidden_size, n_tags, p_drop):
        super(SequenceTagging, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(hidden_size, n_tags)
        self.loss = MaskedCrossEntropyLoss(n_tags)

    def forward(self, features, labels, mask):
        x = self.dropout(features)
        x = self.classifier(x)

        loss = self.loss(x, labels, mask)

        return x, loss


class BinaryTagging(nn.Module):
    def __init__(self, hidden_size, p_drop):
        super(BinaryTagging, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.proj = nn.Linear(hidden_size, 1)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, features, labels, mask):
        x = self.dropout(features)
        x = self.proj(x).squeeze(-1)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mask)
        loss = loss_fct(x, labels)
        accuracy = self.accuracy(x.sigmoid(), labels == 1)
        return x, loss, accuracy


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, n_labels, *args, **kwargs):
        super(MaskedCrossEntropyLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(*args, **kwargs)
        self.n_labels = n_labels

    def forward(self, logits, labels, mask):
        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, self.n_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(self.ce_loss.ignore_index).type_as(labels)
        )

        loss = self.ce_loss(active_logits, active_labels)
        return loss
