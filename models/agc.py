import torch
from pytorch_lightning import LightningModule
from models.extractor import QuestionEncoder, NamedEntityRecognition, QuestionTargetRecognition, AnswerTypeClassification
from models.solver import TemplateSolver


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
        self.template_solver = TemplateSolver(hidden_size, p_drop, language_model.config)
        self.classify_answer_type = AnswerTypeClassification(hidden_size, p_drop, self.template_solver.n_solvers)

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

        answer_types, answer_type_loss, answer_type_accuracy = self.classify_answer_type(batch, features)
        solve_outputs, solve_loss, solve_accuracy = self.template_solver(batch, features, answer_types)

        if qtr_loss and answer_type_loss and solve_loss:
            loss = qtr_loss + answer_type_loss + solve_loss
            accuracy = (qtr_accuracy + answer_type_accuracy + solve_accuracy) / 3

            return qtr_outputs, loss, accuracy
        else:
            return qtr_outputs, None, None

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