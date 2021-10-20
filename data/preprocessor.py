import random
import torch
from data.numerics import NumericProcessor


class Batch(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return None


class Preprocessor:
    __collate_slots = 'input_ids', 'token_type_ids', 'attention_mask', 'equation_type', 'numeric_feature_masks'

    def __init__(self, tokenizer, max_seq_len, wrap_numeric):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.wrap_numeric = wrap_numeric
        self.numeric_processor = NumericProcessor("[NUM]", "[NUMS]")
        custom_tokens = {token: token_id for token, token_id in zip(tokenizer.additional_special_tokens,
                                                                    tokenizer.additional_special_tokens_ids)}
        self.num_token_id = custom_tokens['[NUM]']
        self.nums_token_id = custom_tokens['[NUMS]']

        self._endings = tokenizer.convert_tokens_to_ids(['.', '?'])

    def __call__(self, batch):
        # Merge keys in batch
        batch = Batch({key: [d[key] for d in batch] for key in batch[0]})

        # Replace numeric tokens
        batch['question'], batch['numerics'] = self.numeric_processor.replace_batch(batch['question'])

        # Add Tokenized results
        batch.update(self.tokenizer(batch['question'], padding='longest', truncation=True))

        # Sequence Length
        max_seq_len = len(batch['input_ids'][0])
        seq_lens = [next((i for i, s in enumerate(seq_ids) if s == 0), max_seq_len) for seq_ids in batch['input_ids']]

        # Separator mask
        # batch['sequence_mask'] = batch['attention_mask']
        batch['question_ids'] = batch['input_ids']
        # batch['separator_mask'] = [[int(token_id in self._endings) for token_id in seq_ids] for seq_ids in batch['input_ids']]
        separator_mask = [[int(token_id in self._endings) for token_id in seq_ids] for seq_ids in batch['input_ids']]
        batch['separator_pos'] = [[i for i, m in enumerate(seq) if m] + [0] for seq in separator_mask]

        # Wrap Numeric Mask
        # num_indices = self._wrap_masks([[int(token_id == self.num_token_id) for token_id in seq_ids] for seq_ids in batch['input_ids']])
        num_wrap_ind, num_pos_mask, num_attn_mask = self._wrap(
            [[i for i, token_id in enumerate(seq_ids) if token_id == self.num_token_id] for seq_ids in batch['input_ids']], seq_lens)
        nums_wrap_ind, nums_pos_mask, nums_attn_mask = self._wrap(
            [[i for i, token_id in enumerate(seq_ids) if token_id == self.nums_token_id] for seq_ids in batch['input_ids']], seq_lens)

        batch['num_wrap_ind'] = [torch.as_tensor(n) for n in num_wrap_ind]
        batch['num_pos_mask'] = [torch.as_tensor(m) for m in num_pos_mask]
        batch['num_attn_mask'] = [torch.as_tensor(m) for m in num_attn_mask]
        batch['nums_wrap_ind'] = [torch.as_tensor(n) for n in nums_wrap_ind]
        batch['nums_pos_mask'] = [torch.as_tensor(m) for m in nums_pos_mask]
        batch['nums_attn_mask'] = [torch.as_tensor(m) for m in nums_attn_mask]

        return {k: _collate(v) if k in self.__collate_slots else v for k, v in batch.items()}

    def _wrap(self, batch_indices, seq_lens):
        w = self.wrap_numeric
        size = w * 2 + 1
        wrap_inds = [[list(range(start := max(min(i + w + 1, seq_len) - size, 1), start + size)) for i in indices]
                     for indices, seq_len in zip(batch_indices, seq_lens)]
        pos_masks = [[[(1 if ind == wi else 0) for wi in wis]
                      for ind, wis in zip(indices, wrap_ind)]
                     for wrap_ind, indices in zip(wrap_inds, batch_indices)]
        attn_masks = [[[(0 if wi != ind and wi in indices else 1) for wi in wis]
                       for ind, wis in zip(indices, wrap_ind)]
                      for wrap_ind, indices in zip(wrap_inds, batch_indices)]
        return wrap_inds, pos_masks, attn_masks


class TrainingPreprocessor(Preprocessor):
    def __init__(self, tokenizer, max_seq_len, wrap_numeric):
        super(TrainingPreprocessor, self).__init__(tokenizer, max_seq_len, wrap_numeric)

        self._question_suffixes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("입니까 얼마입니까 구하시오 인가 됩니까 쓰시오 될까요 할까요"))

    def __call__(self, data):
        # Shuffle question and body with [SEP] token
        sep = ' '
        question_pos = []
        for d in data:
            body, question = d.pop('origin_body'), d.pop('origin_question')
            t_body, t_question = d.pop('token_body'), d.pop('token_question')

            sentences = [sb for b in _split_seq(body) if (sb := b.strip())] + [question]
            t_sentences = [sb for b in _split_seq(t_body) if (sb := b.strip())] + [t_question]

            random.shuffle(z := list(zip(sentences, t_sentences)))
            sentences, t_sentences = zip(*z)

            question_pos.append(sentences.index(question))
            d['question'] = sep.join(sentences)
            d['t_question'] = sep.join(t_sentences)

        batch = super(TrainingPreprocessor, self).__call__(data)

        # Map Numerics with Token
        batch['matched_num'] = []
        batch['matched_nums'] = []
        for question, t_question, tokens in zip(batch['question'], batch['t_question'], batch['tokens']):
            # Calculate ltrs = list of (left, token, right) for each numeric token
            qs = [(q[1:], '[NUM]') if q.startswith(']') else (q[2:], '[NUMS]') if q.startswith('S]') else (q, None) for q in question.split('[NUM')]
            ltrs = [(_split_seq(qs[i][0])[-1].strip(), qs[i + 1][1], _split_seq(qs[i + 1][0])[0].strip()) for i in range(len(qs) - 1)]

            t_pos = 0
            # matched_tokens = []
            num_tokens = {}
            nums_tokens = {}
            for left, token, right in ltrs:
                lp = t_question.find(left, t_pos) if left else t_pos
                rp = t_question.find(right, t_pos) if right else t_pos
                if not (rp > (lp := lp + len(left)) >= 0):
                    raise ValueError(f"err on parsing : {lp=} {rp=} {left=}, {right=}, {t_question[t_pos]=}, {t_pos=}, {t_question=}")

                target = t_question[lp:rp]
                matched_token = next((k for k in reversed(tokens.keys()) if k in target), None)
                if not matched_token:
                    raise ValueError(f"err on matching : {lp=} {rp=} {left=}, {right=}, {target=} {t_question[t_pos]=}, {t_pos=}, {t_question=}")

                if token == '[NUM]':
                    num_tokens[matched_token] = tokens[matched_token]
                elif token == '[NUMS]':
                    nums_tokens[matched_token] = tokens[matched_token]
                # matched_tokens.append((token, matched_token, tokens[matched_token]))
                t_pos = rp

            batch['matched_num'].append(num_tokens)
            batch['matched_nums'].append(nums_tokens)

        # Match Equation Targets
        batch['equation_targets'] = []
        for equation_tokens, matched_num, matched_nums in zip(batch['equation_tokens'], batch['matched_num'], batch['matched_nums']):
            equation_target = []
            for token in equation_tokens:
                if token in matched_num:
                    equation_target.append(torch.tensor([list(matched_num).index(token)]))
                elif token in matched_nums:
                    equation_target.append(torch.tensor([list(matched_nums).index(token)]))
                else:
                    equation_target.append(torch.as_tensor(token))
            batch['equation_targets'].append(equation_target)

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
