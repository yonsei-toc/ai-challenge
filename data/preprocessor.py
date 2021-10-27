import random
import re
import difflib
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

    def __init__(self, tokenizer, wrap_numeric):
        self.tokenizer = tokenizer
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
        batch = {key: [d[key] for d in batch] for key in batch[0]}

        # Replace name tokens
        batch['question'], batch['names'] = self.naming_processor.replace_batch(batch['question'])

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

        raws = {k: batch[k] for k in self.__collate_slots if k in batch}
        batch = Batch({k: _collate(v) if k in self.__collate_slots else v for k, v in batch.items()})
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
    def __init__(self, tokenizer, wrap_numeric, injection_prob=0.5):
        super(TrainingPreprocessor, self).__init__(tokenizer, wrap_numeric)

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
            if data[i]['equation_type'] == data[j]['equation_type']:
                continue

            d['inject_sentence'] = random.choice(data[j]['origin_sentences'][:-1])
        for d in data:
            if inject_sentence := d.pop('inject_sentence', None):
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
            d['origin_question'] = d['question'] = sep.join(sentences)
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

            num_tokens = []
            nums_tokens = []
            name_tokens = {}
            all_ts = set()
            iter_t_ltrs = iter(t_ltrs)

            t_left, t_token, t_right = t_next = next(iter_t_ltrs)
            first = True

            sm_left = difflib.SequenceMatcher(None)
            sm_right = difflib.SequenceMatcher(None)
            flag_break = False
            for n_left, n_token, n_right in q_ltrs:
                if n_token == '[NUM]':
                    num_tokens.append(None)
                elif n_token == '[NUMS]':
                    nums_tokens.append(None)
                if flag_break:
                    continue
                sm_left.set_seqs(t_left, n_left)
                sm_right.set_seqs(t_right, n_right)
                left_match = sm_left.find_longest_match(0, len(t_left), 0, len(n_left))
                right_match = sm_right.find_longest_match(0, len(t_right), 0, len(n_right))

                if ((first and t_left == n_left) or
                    (not first and left_match.b + left_match.size == len(n_left) and
                     t_left.endswith(n_left[left_match.b:]))
                ) and ((right_match.b <= 6 and right_match.b + right_match.size == len(n_right) and
                        t_right.startswith(n_right[right_match.b:])) or
                       (right_match.a <= 6 and right_match.b == 0 and right_match.size == len(n_right))
                ):
                    if n_token == '[NUM]':
                        num_tokens[-1] = t_token
                    elif n_token == '[NUMS]':
                        nums_tokens[-1] = t_token
                    elif n_token.startswith('[NAME') and n_token.endswith(']'):
                        name_tokens[t_token] = n_token
                    else:
                        raise ValueError(f"Not token : {n_token}")
                    all_ts.add(t_token)
                    if (t_next := next(iter_t_ltrs, None)) is None:
                        flag_break = True
                        continue
                    t_left, t_token, t_right = t_next
                first = False

            if t_next is not None:
                raise ValueError(f"Error on matching : {t_next=} {num_tokens=} {nums_tokens=} {name_tokens=} {tokens=} {t_question=} {question=}")
            if set(tokens.keys()) - all_ts:
                raise ValueError(f"Error on matching : {t_next=} {num_tokens=} {nums_tokens=} {name_tokens=} {tokens=} {t_question=} {question=}")

            batch['matched_num'].append(num_tokens)
            batch['matched_nums'].append(nums_tokens)
            batch['matched_names'].append(name_tokens)

        # Match Equation Targets
        batch['equation_targets'] = []
        for i, (eq_tokens, matched_num, matched_nums, eq_type) in enumerate(zip(batch['equation_tokens'], batch['matched_num'], batch['matched_nums'],
                                                                                raw_batch['equation_type'])):
            if eq_type == 4:
                equation_target = self._make_order_by_comp_target(i, batch, raw_batch)
            elif eq_type in (6, 11, 12):
                equation_target = self._make_sum_num_sig_target(i, batch, raw_batch)
            elif eq_type in (8, 13, 14):
                equation_target = self._make_max_sub_min2_target(i, batch, raw_batch)
            elif eq_type == 10:
                equation_target = self._make_count_from_compare_pivot2(i, batch, raw_batch)
            else:
                equation_target = []
                for token in eq_tokens:
                    if token in matched_num:
                        equation_target.append(torch.tensor([matched_num.index(token)]))
                    elif token in matched_nums:
                        equation_target.append(torch.tensor([matched_nums.index(token)]))
                    elif isinstance(token, str):
                        raise ValueError(token)
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

    # Make OrderByComp Answer
    def _make_order_by_comp_target(self, batch_idx, batch, raw_batch):
        eq_token = batch['equation_tokens'][batch_idx][0]
        seq_ids = raw_batch['input_ids'][batch_idx]
        matched_names = batch['matched_names'][batch_idx]

        eq_name = matched_names[eq_token]
        eq_token_id = self.custom_tokens[eq_name]
        equation_target = [float(seq_id == eq_token_id) for seq_id in seq_ids]

        return torch.as_tensor(equation_target)

    # Make SumNumSig Answer
    def _make_sum_num_sig_target(self, batch_idx, batch, raw_batch):
        eq_token = batch['equation_args'][batch_idx]
        matched_num = batch['matched_num'][batch_idx]
        equation_target = []
        for num in matched_num:
            if num is None:
                equation_target.append(0)
                continue
            target_token = next((t for t in eq_token if t.token == num), None)
            if target_token.sgn >= 0:
                equation_target.append(target_token.sgn)
            else:
                equation_target.append(2)
        return torch.as_tensor(equation_target)

    def _make_max_sub_min2_target(self, batch_idx, batch, raw_batch):
        eq_token = batch['equation_tokens'][batch_idx]
        matched_num = batch['matched_num'][batch_idx]
        equation_target = [0. if num is None or num not in eq_token else 1. for num in matched_num]
        return torch.as_tensor(equation_target)

    def _make_count_from_compare_pivot2(self, batch_idx, batch, raw_batch):
        eq_token = batch['equation_tokens'][batch_idx]
        matched_num = batch['matched_num'][batch_idx]
        equation_target0 = torch.tensor([eq_token[0]])
        equation_target1 = torch.tensor([matched_num.index(eq_token[1])])
        equation_target2 = [0. if num is None or num not in eq_token[2:] else 1. for num in matched_num]
        return equation_target0, equation_target1, torch.as_tensor(equation_target2)


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
