import random
import re
import torch
from data.numerics import NumericProcessor
from data.naming import NamingProcessor


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
        self.naming_processor = NamingProcessor()
        self.custom_tokens = {token: token_id for token, token_id in zip(tokenizer.additional_special_tokens,
                                                                         tokenizer.additional_special_tokens_ids)}
        self.num_token_id = self.custom_tokens['[NUM]']
        self.nums_token_id = self.custom_tokens['[NUMS]']
        self.name_token_ids = {v: k for k, v in self.custom_tokens.items() if k not in ('[NUM]', '[NUMS]')}

        self._endings = tokenizer.convert_tokens_to_ids(['.', '?'])

    def __call__(self, batch):
        # Merge keys in batch
        batch = Batch({key: [d[key] for d in batch] for key in batch[0]})

        # Replace numeric tokens
        batch['question'], batch['numerics'] = self.numeric_processor.replace_batch(batch['question'])

        # Replace name tokens
        batch['question'], batch['names'] = self.naming_processor.replace_batch(batch['question'])

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
        self.search_token = re.compile("(\\#\\d)")
        self.search_specials = re.compile('(' + '|'.join([re.escape(k) for k in self.custom_tokens]) + ')')

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

        # Tokenize
        batch = super(TrainingPreprocessor, self).__call__(data)
        raw_batch = batch.pop('raw_collated')

        # Map Special Tokens
        batch['matched_num'] = []
        batch['matched_nums'] = []
        batch['matched_names'] = []
        for question, t_question, tokens in zip(batch['question'], batch['t_question'], batch['tokens']):
            q_splits = self.search_specials.split(question)
            t_splits = self.search_token.split(t_question)

            q_ltrs = [(q_splits[i - 1].strip(), q_splits[i], q_splits[i + 1].strip()) for i in range(1, len(q_splits) - 1, 2)]
            t_ltrs = [(t_splits[i - 1].strip(), t_splits[i], t_splits[i + 1].strip()) for i in range(1, len(t_splits) - 1, 2)]

            if not t_ltrs:
                continue

            num_tokens = {}
            nums_tokens = {}
            name_tokens = {}
            iter_t_ltrs = iter(t_ltrs)

            t_left, t_token, t_right = t_next = next(iter_t_ltrs)

            for n_left, n_token, n_right in q_ltrs:
                if t_left.endswith(n_left) and t_right.startswith(n_right):
                    if n_token == '[NUM]':
                        num_tokens[t_token] = tokens[t_token]
                    elif n_token == '[NUMS]':
                        nums_tokens[t_token] = tokens[t_token]
                    elif n_token.startswith('[NAME') and n_token.endswith(']'):
                        name_tokens[t_token] = n_token
                    if (t_next := next(iter_t_ltrs, None)) is None:
                        break
                    t_left, t_token, t_right = t_next

            while t_next is not None:
                t_left, t_token, t_right = t_next
                lp, lsub = next(((lp, t_left[i:]) for i in range(len(t_left) - 1) if (lp := question.find(t_left[i:])) >= 0), (-1, ''))
                rp, rsub = next(((rp, t_right[i:]) for i in range(len(t_right) - 1) if (rp := question.find(t_right[i:])) >= 0), (-1, ''))
                lp += len(lsub)
                if rp > lp >= 0 and lsub and rsub:
                    raise ValueError(f"Not Error But check case : {lp=} {rp=} {question[lp:rp]=} {lsub=} {rsub=}"
                                     f"{t_next=} {num_tokens=} {nums_tokens=} {name_tokens=} {tokens=} {t_question=} {question=}")
                else:
                    raise ValueError(f"Extra Matching Error : {lp=} {rp=} {question[lp:rp]=} {lsub=} {rsub=}"
                                     f"{t_next=} {num_tokens=} {nums_tokens=} {name_tokens=} {tokens=} {t_question=} {question=}")

                t_next = next(iter_t_ltrs, None)
            if t_next is not None:
                raise ValueError(f"Error on matching : {t_next=} {num_tokens=} {nums_tokens=} {name_tokens=} {tokens=} {t_question=} {question=}")

            batch['matched_num'].append(num_tokens)
            batch['matched_nums'].append(nums_tokens)
            batch['matched_names'].append(name_tokens)

        # Match Equation Targets
        batch['equation_targets'] = []
        for eq_tokens, matched_num, matched_nums, eq_type in zip(batch['equation_tokens'], batch['matched_num'], batch['matched_nums'],
                                                                 raw_batch['equation_type']):
            equation_target = []
            for token in eq_tokens:
                if token in matched_num:
                    equation_target.append(torch.tensor([list(matched_num).index(token)]))
                elif token in matched_nums:
                    equation_target.append(torch.tensor([list(matched_nums).index(token)]))
                elif isinstance(token, str):
                    pass
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
