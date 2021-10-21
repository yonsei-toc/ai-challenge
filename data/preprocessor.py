import random
import torch
from data.numerics import NumericProcessor


class Batch(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return None


class Preprocessor:
    __collate_slots = 'input_ids', 'token_type_ids', 'attention_mask', 'equation_type', 'unnum_mask'

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

        # Wrap Numeric Mask
        num_wrap_ind, num_pos_mask, num_attn_mask = self._wrap(
            [[i for i, token_id in enumerate(seq_ids) if token_id == self.num_token_id] for seq_ids in batch['input_ids']], seq_lens)
        nums_wrap_ind, nums_pos_mask, nums_attn_mask = self._wrap(
            [[i for i, token_id in enumerate(seq_ids) if token_id == self.nums_token_id] for seq_ids in batch['input_ids']], seq_lens)
        batch['unnum_mask'] = [[(m if t != self.num_token_id and t != self.nums_token_id else 0) for t, m in zip(seq_ids, masks)]
                               for seq_ids, masks in zip(batch['input_ids'], batch['attention_mask'])]

        batch['num_wrap_ind'] = [torch.as_tensor(n) for n in num_wrap_ind]
        batch['num_pos_mask'] = [torch.as_tensor(m) for m in num_pos_mask]
        batch['num_attn_mask'] = [torch.as_tensor(m) for m in num_attn_mask]
        batch['nums_wrap_ind'] = [torch.as_tensor(n) for n in nums_wrap_ind]
        batch['nums_pos_mask'] = [torch.as_tensor(m) for m in nums_pos_mask]
        batch['nums_attn_mask'] = [torch.as_tensor(m) for m in nums_attn_mask]

        raws = {k: batch[k] for k in self.__collate_slots}
        batch = {k: _collate(v) if k in self.__collate_slots else v for k, v in batch.items()}
        batch['raw_collated'] = raws

        return batch

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
    def __init__(self, tokenizer, max_seq_len, wrap_numeric, injection_prob=0.5):
        super(TrainingPreprocessor, self).__init__(tokenizer, max_seq_len, wrap_numeric)

        self.injection_prob = injection_prob
        self._question_suffixes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("입니까 얼마입니까 구하시오 인가 됩니까 쓰시오 될까요 할까요"))

    def __call__(self, data):
        # Split body and question into each sentences
        question_pos = []
        for d in data:
            body, question = d.pop('origin_body'), d.pop('origin_question')
            t_body, t_question = d.pop('token_body'), d.pop('token_question')

            d['origin_sentences'] = [sb for b in _split_seq(body) if (sb := b.strip())] + [question]
            d['token_sentences'] = [sb for b in _split_seq(t_body) if (sb := b.strip())] + [t_question]

        # Inject random sentences of body from other body of batch
        for i, d in enumerate(data):
            if random.random() > self.injection_prob:
                continue
            if i == (j := random.randrange(0, len(data))):
                continue
            if len(data[j]['origin_sentences']) == 1:
                continue

            inject_sentence = random.choice(data[j]['origin_sentences'][:-1])
            d['origin_sentences'].insert(0, inject_sentence)
            d['token_sentences'].insert(0, inject_sentence)

        # Shuffle question and body with [SEP] token
        sep = ' '
        for d in data:
            sentences, t_sentences = d.pop('origin_sentences'), d.pop('token_sentences')
            question = sentences[-1]

            random.shuffle(z := list(zip(sentences, t_sentences)))
            sentences, t_sentences = zip(*z)

            question_pos.append(sentences.index(question))
            d['question'] = sep.join(sentences)
            d['t_question'] = sep.join(t_sentences)

        batch = super(TrainingPreprocessor, self).__call__(data)
        raw_batch = batch.pop('raw_collated')

        # Map Numerics with Token
        batch['matched_num'] = []
        batch['matched_nums'] = []
        for question, t_question, tokens in zip(batch['question'], batch['t_question'], batch['tokens']):
            # Calculate ltrs = list of (left, token, right) for each numeric token
            qs = [(q[1:], '[NUM]') if q.startswith(']') else (q[2:], '[NUMS]') if q.startswith('S]') else (q, None) for q in question.split('[NUM')]
            ltrs = [(_split_seq(qs[i][0])[-1].strip(), qs[i + 1][1], _split_seq(qs[i + 1][0])[0].strip()) for i in range(len(qs) - 1)]

            t_pos = 0
            num_tokens = {}
            nums_tokens = {}
            for left, token, right in ltrs:
                lp = (t_question.find(left, t_pos) + len(left)) if left else t_pos
                rp = t_question.find(right, lp) if right else t_pos

                target = t_question[lp:rp]
                matched_token = next((k for k in reversed(tokens.keys()) if k in target), None)
                if matched_token:
                    if token == '[NUM]':
                        num_tokens[matched_token] = tokens[matched_token]
                    elif token == '[NUMS]':
                        nums_tokens[matched_token] = tokens[matched_token]
                t_pos = rp
            all_keys = set(num_tokens.keys()).union(set(nums_tokens.keys()))
            if all_keys != set(tokens.keys()):
                raise ValueError(f"Error on matching : {num_tokens=} {nums_tokens=} {tokens=} {t_question=} {question=}")

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
                    t = torch.as_tensor(token)
                    if len(t.shape) == 0:
                        t = t.unsqueeze(-1)
                    equation_target.append(t)
            batch['equation_targets'].append(equation_target)

        # Make Question Targets
        separator_mask = [[int(token_id in self._endings) for token_id in seq_ids] for seq_ids in raw_batch['input_ids']]
        separator_pos = [[i for i, m in enumerate(seq) if m] + [0] for seq in separator_mask]

        length = len(raw_batch['input_ids'][0])
        question_targets = []
        for qp, sp, qids in zip(question_pos, separator_pos, raw_batch['input_ids']):
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
