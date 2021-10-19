import random
import torch
from data.numerics import NumericProcessor


class Preprocessor:
    __collate_slots = 'input_ids', 'token_type_ids', 'attention_mask', 'equation_type'

    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.numeric_processor = NumericProcessor("[NUM]", "[NUMS]")

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
        sep = ' '
        question_pos = []
        for d in batch:
            body, question = d.pop('origin_body'), d.pop('origin_question')
            t_body, t_question = d.pop('token_body'), d.pop('token_question')

            sentences = [sb for b in _split_seq(body) if (sb := b.strip())] + [question]
            t_sentences = [sb for b in _split_seq(t_body) if (sb := b.strip())] + [t_question]

            random.shuffle(z := list(zip(sentences, t_sentences)))
            sentences, t_sentences = zip(*z)

            question_pos.append(sentences.index(question))
            d['question'] = sep.join(sentences)
            d['t_question'] = sep.join(t_sentences)

        batch = super(TrainingPreprocessor, self).__call__(batch)

        # Map Numerics with Token
        batch['matched_tokens'] = []
        for question, t_question, tokens in zip(batch['question'], batch['t_question'], batch['tokens']):
            # Calculate ltrs = list of (left, token, right) for each numeric token
            qs = [(q[1:], '[NUM]') if q.startswith(']') else (q[2:], '[NUMS]') if q.startswith('S]') else (q, None) for q in question.split('[NUM')]
            ltrs = [(_split_seq(qs[i][0])[-1].strip(), qs[i + 1][1], _split_seq(qs[i + 1][0])[0].strip()) for i in range(len(qs) - 1)]

            t_pos = 0
            matched_tokens = []
            for left, token, right in ltrs:
                lp = t_question.find(left, t_pos) if left else t_pos
                rp = t_question.find(right, t_pos) if right else t_pos
                if not (rp > (lp := lp + len(left)) >= 0):
                    raise ValueError(f"err on parsing : {lp=} {rp=} {left=}, {right=}, {t_question[t_pos]=}, {t_pos=}, {t_question=}")

                target = t_question[lp:rp]
                matched_token = next((k for k in reversed(tokens.keys()) if k in target), None)
                if not matched_token:
                    raise ValueError(f"err on matching : {lp=} {rp=} {left=}, {right=}, {target=} {t_question[t_pos]=}, {t_pos=}, {t_question=}")

                matched_tokens.append((token, matched_token, tokens[matched_token]))
                t_pos = rp

            batch['matched_tokens'].append(matched_tokens)

        # Make Question Targets
        length = len(batch['question_ids'][0])
        question_targets = []
        for qp, sp, qids in zip(question_pos, batch['separator_pos'], batch['question_ids']):
            qt = [0.0] * length
            for i in range(sp[qp - 1] + 1, sp[qp]):
                qt[i] = 1.0
                if qids[i] in self._question_suffixes:
                    break
            question_targets.append(qt)
        batch['question_targets'] = _collate(question_targets)

        return batch


def _collate(batch):
    elem = batch[0]
    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, list):
        try:
            return torch.as_tensor(batch)
        except ValueError:
            return batch
    elif isinstance(elem, tuple):
        return batch
    else:
        raise TypeError(f"wrong type of collate batch : {type(batch)}")


def _split_seq(s):
    return s.replace('다.', '다.||').replace('요.', '요.||').split('||')
