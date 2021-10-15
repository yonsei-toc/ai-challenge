import random
import torch
from data.numerics import NumericProcessor


class Preprocessor:
    __collate_slots = 'input_ids', 'token_type_ids', 'attention_mask', 'equation_type'

    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.numeric_processor = NumericProcessor('[NUM]')

        self._endings = tokenizer.convert_tokens_to_ids(['.', '?'])

    def __call__(self, batch):
        # Merge keys in batch
        batch = {key: [d[key] for d in batch] for key in batch[0]}

        # Replace numeric tokens
        batch['question'], batch['numerics'] = self.numeric_processor.replace_batch(batch['question'])

        # Add Tokenized results
        batch.update(self.tokenizer(batch['question'], padding='longest', truncation=True))

        # Separator mask
        batch['sequence_mask'] = batch['attention_mask']
        batch['question_ids'] = batch['input_ids']
        batch['separator_mask'] = [[int(token_id in self._endings) for token_id in seq_ids] for seq_ids in batch['input_ids']]
        batch['separator_pos'] = [[i for i, m in enumerate(seq) if m] + [0] for seq in batch['separator_mask']]

        return {k: _collate(v) if k in self.__collate_slots else v for k, v in batch.items()}


class TrainingPreprocessor(Preprocessor):
    def __init__(self, tokenizer, max_seq_len):
        super(TrainingPreprocessor, self).__init__(tokenizer, max_seq_len)

        self._question_suffixes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("입니까 얼마입니까 구하시오 인가 됩니까 쓰시오 될까요 할까요"))

    def __call__(self, batch):
        # Shuffle question and body with [SEP] token
        sep = ' '  # f" {self.tokenizer.sep_token} "
        question_pos = []
        for d in batch:
            body, question = d.pop('body'), d.pop('question')

            sentences = [sb for b in body.replace('다.', '다.||').replace('요.', '요.||').split('||') if len(sb := b.strip()) > 0]
            sentences.append(question)

            random.shuffle(sentences)

            question_pos.append(sentences.index(question))
            d['question'] = sep.join(sentences)

        batch = super(TrainingPreprocessor, self).__call__(batch)

        # Make Question Targets
        question_targets = [([0] * (s[q - 1] + 1) + [1.] * (s[q] - s[q - 1] - 1) + [0] * (len(batch['question_ids'][0]) - s[q])) for q, s in zip(
            question_pos, batch['separator_pos'])]
        for qts, qids in zip(question_targets, batch['question_ids']):
            for i, qt, qid in zip(range(len(qts)), qts, qids):
                if qt != 1:
                    continue

                if qid in self._question_suffixes:
                    qts[i:] = [0] * (len(qts) - i)
                    break
        batch['question_targets'] = _collate(question_targets)

        return batch


def _collate(batch):
    elem = batch[0]
    for b in batch:
        if type(elem) != type(b):
            return batch

    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, list):
        # if len(elem) == 0 or isinstance(elem[0], tuple):
        #     return batch
        try:
            return torch.as_tensor(batch)
        except ValueError:
            return batch
    elif isinstance(elem, tuple):
        return batch
    else:
        raise TypeError(f"wrong type of collate batch : {type(batch)}")
