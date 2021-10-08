import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    def __init__(self, language_model):
        super(QuestionEncoder, self).__init__()
        self.language_model = language_model


class NamedEntityRecognition(nn.Module):
    def __init__(self):
        super(NamedEntityRecognition, self).__init__()


class SentenceClassifier(nn.Module):
    def __init__(self):
        super(SentenceClassifier, self).__init__()

    def forward(self, batch, features):
        pass


class SequenceClassifier(nn.Module):
    def __init__(self, hidden_size, n_labels, p_drop):
        super(SequenceClassifier, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.proj = nn.Linear(hidden_size, n_labels)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, features, labels, mask):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)

        active_loss = mask.view(-1) == 1
        active_logits = x.view(-1, self.config.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
        )
        loss = self.loss(active_logits, active_labels)
        return x


class TemplateSolver(nn.Module):
    def __init__(self):
        super(TemplateSolver, self).__init__()


class SequenceTagging(nn.Module):
    def __init__(self, hidden_size, n_tags, p_drop):
        super(SequenceTagging, self).__init__()
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(hidden_size, n_tags)

    def forward(self, batch, features):
        x = self.dropout(features)
        x = self.classifier(x)
        return x
