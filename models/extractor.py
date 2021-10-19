import torch.nn as nn

import models.base as base


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


class AnswerTypeClassification(nn.Module):
    def __init__(self, hidden_size, p_drop, n_types):
        super(AnswerTypeClassification, self).__init__()
        self.classifier = base.SequenceClassifier(hidden_size, n_types, p_drop)

    def forward(self, batch, features):
        outputs, loss, accuracy = self.classifier(features, batch['equation_type'])
        answer_types = outputs.argmax(-1)

        return answer_types, loss, accuracy
