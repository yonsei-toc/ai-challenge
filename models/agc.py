import torch
from pytorch_lightning import LightningModule
from transformers import ElectraModel
from models.extractor import NamedEntityRecognition, QuestionTargetRecognition, AnswerTypeClassification
from models.solver import TemplateSolver
from data.equations import equations


class AGCModel(LightningModule):
    def __init__(self, language_model=None, tokenizer=None, learning_rate=5e-5, p_drop=0.1):
        super(AGCModel, self).__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        print(f"AGC Model()\n{self.hparams}")

        language_model = ElectraModel.from_pretrained(language_model)
        language_model.resize_token_embeddings(len(tokenizer.vocab))
        self.id_to_token = {token_id: token for token, token_id in zip(tokenizer.additional_special_tokens,
                                                                       tokenizer.additional_special_tokens_ids)}
        self.token_to_id = {token: token_id for token, token_id in zip(tokenizer.additional_special_tokens,
                                                                       tokenizer.additional_special_tokens_ids)}
        self.tokenizer = tokenizer

        hidden_size = language_model.config.hidden_size

        self.learning_rate = learning_rate

        self.language_model = language_model
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

    def get_action_results(self, batch, tag):
        features = self(batch)

        # Prepare for solving
        # ner_outputs, ner_loss = self.ner(batch, features)
        qtr_outputs, qtr_loss, qtr_accuracy = self.qtr(batch, features)
        answer_types, answer_type_loss, answer_type_accuracy = self.classify_answer_type(batch, features)

        # Solve Questions
        if 'question_targets' in batch:
            question_mask = batch['question_targets'].int()
        else:
            question_mask = (qtr_outputs >= 0.5).int()
        question_mask = (question_mask * batch['unnum_mask']).int()

        solve_outputs, solve_loss, solve_accuracy, solve_results = self.template_solver(batch, features, answer_types, question_mask)

        if qtr_loss and answer_type_loss and solve_loss:
            loss = qtr_loss + answer_type_loss + solve_loss
            accuracy = (qtr_accuracy + answer_type_accuracy + solve_accuracy) / 3

            self.log_dict({
                f"pre_solver({tag})/qtr_loss": qtr_loss,
                f"pre_solver({tag})/qtr_accuracy": qtr_accuracy,
                f"pre_solver({tag})/answer_type_loss": answer_type_loss,
                f"pre_solver({tag})/answer_type_accuracy": answer_type_accuracy
            })
            self.log_dict(solve_results)

            return solve_outputs, loss, accuracy
        else:
            return solve_outputs, None, None

    def training_step(self, batch, batch_idx):
        output, loss, accuracy = self.get_action_results(batch, 'train')

        self.log("train/accuracy", accuracy, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output, loss, accuracy = self.get_action_results(batch, 'val')

        self.log_dict({"valid/loss": loss, "valid/accuracy": accuracy}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (answer_types, model_outputs), _, _ = self.get_action_results(batch, 'predict')
        keys = []
        answers = []
        output_codes = []

        batch_num_list = []
        batch_nums_list = []
        for numerics in batch['numerics']:
            num_list = []
            nums_list = []
            if numerics:
                for numeric in numerics:
                    if isinstance(numeric, list):
                        nums_list.append([n[1] for n in numeric])
                    else:
                        num_list.append(numeric[1])
            batch_num_list.append(num_list)
            batch_nums_list.append(nums_list)
        batch_names = batch['names']
        min_name_id = self.token_to_id['[NAME1]']

        for ans_type, model_output, key, input_ids, num, nums, names in zip(answer_types, model_outputs, batch['key'],
                                                                            batch['input_ids'], batch_num_list, batch_nums_list, batch_names):
            ans_type = ans_type.item()
            answer = 0
            code = 'print(0)  # Failed to solve\n'
            equation_fn = lambda *x: code
            if ans_type >= 0:
                equation_fn = equations.equations[ans_type]
            try:
                if ans_type == 0:  # DiffPerm
                    num_idx = model_output[0]
                    nums_idx = model_output[1]
                    type_idx = model_output[2]
                    code = equation_fn(num[num_idx], nums[nums_idx], type_idx)
                elif ans_type == 1:  # CountFromRange
                    num_idxes = model_output
                    params = [(num[ni] if ni >= 0 else 1) for ni in num_idxes]
                    code = equation_fn(*params)
                elif ans_type == 2:  # FindSumFromRange
                    num_idxes = model_output
                    params = [(num[ni] if ni >= 0 else 1) for ni in num_idxes]
                    code = equation_fn(*params)
                elif ans_type == 3:  # WrongMultiply
                    num_idxes = model_output[:-1]
                    type_idx = model_output[-1]
                    params = [(num[ni] if ni >= 0 else 1) for ni in num_idxes]
                    if len(num) == 6:
                        code = equation_fn(3, *params[1:], type_idx)
                    elif len(num) == 5:
                        code = equation_fn(3, *params[1:], type_idx)
                    elif len(num) == 4:
                        code = equation_fn(3, 1, *params[2:], type_idx)
                    elif len(num) == 3:
                        code = equation_fn(3, 1, 2, *params[3:], type_idx)
                    else:
                        code = equation_fn(3, *params[0:], type_idx)
                elif ans_type == 4:  # OrderByCompare
                    if len(names) > 0:
                        name_mask = input_ids >= min_name_id
                        name_outputs = model_output[name_mask]
                        masked_ids = input_ids[name_mask]
                        if name_outputs.numel() > 0:
                            name_id = masked_ids.gather(0, name_outputs.argmax(-1)).item()
                            token = self.id_to_token[name_id]
                            if token in names:
                                name = names[token]
                            else:
                                name = names[list(names.keys())[-1]]
                            code = _order_by_comp_equation(list(names.values()), name)
                        else:
                            name = names[list(names.keys())[-1]]
                            code = _order_by_comp_equation(list(names.values()), name)
                    else:
                        token_id = input_ids.gather(0, model_output.argmax(-1)).item()
                        token = self.tokenizer.convert_ids_to_tokens(token_id)
                        code = equation_fn(token)
                elif ans_type == 5:  # HalfSub
                    num_idxes = model_output
                    params = [(num[ni] if ni >= 0 else 1) for ni in num_idxes]
                    code = equation_fn(*params)
                elif ans_type == 6:  # SumNumSig
                    if len(num) > 0:
                        sigs = [(m if m >= 0 else -1) for m in model_output.numpy()]
                        params = [s * n for s, n in zip(sigs, num)]
                        code = equation_fn(*params)
                    else:
                        params = []
                        for n in nums:
                            params.extend(n)
                        code = equation_fn(*params)
                elif ans_type == 7:  # MaxSubMin
                    nums_idx = model_output[0]
                    code = equation_fn(nums[nums_idx])
                elif ans_type == 8:  # MaxSubMin2
                    num_mask = (model_output >= 0.5)
                    params = [n for n, m in zip(num, num_mask) if m]
                    if len(params) > 1:
                        code = equation_fn(*params)
                    else:
                        code = equation_fn(*num)
                elif ans_type == 9:  # CountFromComparePivot
                    type_idx = model_output[0]
                    num_idx = model_output[1]
                    nums_idx = model_output[2]
                    if len(nums) > 0:
                        code = equation_fn(type_idx, num[num_idx], nums[nums_idx])
                    else:
                        code = equation_fn(type_idx, num[num_idx], num)
                else:
                    if len(nums) > 0:
                        answer = sum(nums[0])
                        code = f"ans = sum({nums[0]})\nprint(ans)\n"
                    else:
                        answers.append(0)
                        code = "ans = sum([])\nprint(ans)\n"
                if ans_type >= 0:
                    _var = {}
                    exec(code.replace('print(ans)\n', ''), _var)
                    if 'ans' in _var:
                        answer = _var['ans']
            except:
                pass
            keys.append(key)
            answers.append(answer)
            output_codes.append(code)
        return keys, answers, output_codes


def _order_by_comp_equation(names, name):
    return f'ans = {names}[{names.index(name)}]\nprint(ans)\n'
