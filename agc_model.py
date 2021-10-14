import torch
from pytorch_lightning import LightningModule
from models import QuestionEncoder, NamedEntityRecognition, QuestionTargetRecognition, TemplateSolver


class AGCModel(LightningModule):
    def __init__(self, language_model, tokenizer, n_types=8, n_templates=40, learning_rate=5e-5, p_drop=0.1,
                 encoder='simple'):
        super(AGCModel, self).__init__()
        self.save_hyperparameters(ignore=['language_model', 'tokenizer'])
        self.save_hyperparameters({'language_model': language_model.name_or_path})
        print(f"AGC Model()\n{self.hparams}")

        hidden_size = language_model.config.hidden_size

        self.learning_rate = learning_rate

        self.language_model = language_model
        if encoder == 'simple':
            self.encoder = None
        else:
            self.encoder = QuestionEncoder()
        self.ner = NamedEntityRecognition(hidden_size, p_drop)
        self.qtr = QuestionTargetRecognition(hidden_size, p_drop)
        # self.find_question = SentenceClassifier(hidden_size, p_drop, tokenizer.sep_token_id, tokenizer.cls_token_id)
        # self.type_classifier = SequenceClassifier(hidden_size, n_types, p_drop)
        # self.template_classifier = SequenceClassifier(hidden_size, n_templates, p_drop)
        self.template_solver = TemplateSolver()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, batch):
        features = self.language_model(input_ids=batch['input_ids'],
                                       token_type_ids=batch['token_type_ids'],
                                       attention_mask=batch['attention_mask'])[0]
        return features

    def get_action_results(self, batch):
        features = self(batch)
        # ner_outputs, ner_loss = self.ner(batch, features)
        qtr_outputs, qtr_loss, qtr_accuracy = self.qtr(batch, features)

        # question_outputs, question_loss, question_accuracy = self.find_question(batch, features)
        # type_outputs, type_loss = self.type_classifier(features)

        loss = qtr_loss

        return qtr_outputs, loss, qtr_accuracy

    def training_step(self, batch, batch_idx):
        output, loss, accuracy = self.get_action_results(batch)

        self.log("train_acc", accuracy, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output, loss, accuracy = self.get_action_results(batch)

        self.log_dict({"valid_loss": loss, "valid_acc": accuracy}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass
