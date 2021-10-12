import random
import torch
from data.numerics import NumericProcessor


class Preprocessor:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.numeric_processor = NumericProcessor('[NUM]')

    def __call__(self, batch):
        # Merge keys in batch
        batch = {key: [d[key] for d in batch] for key in batch[0]}

        # Replace numeric tokens
        batch['question'], batch['numerics'] = self.numeric_processor.replace_batch(batch['question'])

        tokenized = self.tokenizer(batch['question'], padding='longest', truncation=True)
        batch.update(tokenized)
        return {k: _collate(v) for k, v in batch.items()}


class TrainingPreprocessor(Preprocessor):
    def __call__(self, batch):
        # Shuffle question and body with [SEP] token
        sep = self.tokenizer.sep_token
        for d in batch:
            body, question = d.pop('body'), d.pop('question')
            body = body.replace('다.', '다.' + sep)
            body = body.replace('요.', '요.' + sep)
            body = [bs.strip() for bs in body.split(sep) if len(bs.strip()) > 0]
            question_pos = random.randint(0, len(body))
            body.insert(question_pos, question)
            d['question'] = f" {sep} ".join(body)
            d['question_pos'] = question_pos

        result = super(TrainingPreprocessor, self).__call__(batch)
        return result


def _collate(batch):
    elem = batch[0]

    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, list):
        if len(elem) == 0 or isinstance(elem[0], tuple):
            return batch
        return torch.as_tensor(batch)
    else:
        raise TypeError(f"wrong type of collate batch : {type(batch)}")
