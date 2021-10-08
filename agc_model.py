import torch
from pytorch_lightning import LightningModule
from models import QuestionEncoder, NamedEntityRecognition, SentenceClassifier, SequenceClassifier, TemplateSolver


class AGCModel(LightningModule):
    def __init__(self, language_model, n_types=8, n_templates=40, learning_rate=5e-5, p_drop=0.1):
        super(AGCModel, self).__init__()
        self.save_hyperparameters(ignore='language_model')
        self.save_hyperparameters({'language_model': language_model.name_or_path})
        print(f"AGC Model()\n{self.hparams}")

        hidden_size = language_model.config.hidden_size

        self.learning_rate = learning_rate

        self.encoder = QuestionEncoder(language_model)
        self.ner = NamedEntityRecognition()
        self.find_question = SentenceClassifier()
        self.type_classifier = SequenceClassifier(hidden_size, n_types, p_drop)
        self.template_classifier = SequenceClassifier(hidden_size, n_templates, p_drop)
        self.template_solver = TemplateSolver()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, batch):
        features = self.encoder(input_ids=batch['input_ids'],
                                token_type_ids=batch['token_type_ids'],
                                attention_mask=batch['attention_mask'])
        return features

    def step_action(self, batch, features):
        output = None
        loss = None

        return output, loss

    def training_step(self, batch, batch_idx):
        features = self(batch)

        output, loss = self.step_action(batch, features)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
