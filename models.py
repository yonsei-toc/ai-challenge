import torch
import torch.nn as nn

import base_models as base


class QuestionEncoder(nn.Module):
    def __init__(self):
        super(QuestionEncoder, self).__init__()

    def forward(self, batch, features):
        print(self)
        return None


class NamedEntityRecognition(nn.Module):
    def __init__(self, hidden_size, p_drop):
        super(NamedEntityRecognition, self).__init__()
        self.tag_names = base.SequenceTagging(hidden_size, 3, p_drop)

    def forward(self, batch, features):
        outputs, loss = self.tag_names(features, labels, mask)

        return outputs, loss


class QuestionTargetRecognition(nn.Module):
    def __init__(self, hidden_size, p_drop):
        super(QuestionTargetRecognition, self).__init__()
        self.tag_target = base.BinaryTagging(hidden_size, p_drop)

    def forward(self, batch, features):
        outputs, loss, accuracy = self.tag_target(features, batch['question_targets'], batch['attention_mask'])

        return outputs, loss, accuracy


class TemplateSolver(nn.Module):
    def __init__(self):
        super(TemplateSolver, self).__init__()

    def forward(self, batch):
        pass
