import itertools
import torch
import torch.nn as nn

import models.base as base


class TemplateSolver(base.Module):
    def __init__(self, hidden_size, p_drop, config):
        super(TemplateSolver, self).__init__()
        self.solvers = nn.ModuleList([
            _DiffPerm(hidden_size, p_drop),
            _CountFromRange(hidden_size, p_drop),
            _FindSumFromRange(hidden_size, p_drop),
            _WrongMultiply(hidden_size, p_drop),
            _OrderByCompare(hidden_size, p_drop, config),
            _HalfSub(hidden_size, p_drop),
            _SumNumSig(hidden_size, p_drop),
            _MaxSubMin(hidden_size, p_drop),
            _MaxSubMin2(hidden_size, p_drop),
            _CountFromComparePivot(hidden_size, p_drop),
            _CountFromComparePivot2(hidden_size, p_drop)
        ])
        self.extract_num = TokenFeatureExtractor('num', config)
        self.extract_nums = TokenFeatureExtractor('nums', config)
        self.n_solvers = len(self.solvers)

    def fetch_all(self, batch, answer_types, features, num_features, nums_features):
        label_answer_type = batch['equation_type']
        fetched = []
        if label_answer_type is None:
            input_ids = batch['input_ids']
            solver_ids = []
            for bi, answer_type in enumerate(answer_types):
                n_num = 0 if num_features[bi] is None else num_features[bi].size(0)
                n_nums = 0 if nums_features[bi] is None else nums_features[bi].size(0)
                n_names = (input_ids[bi] >= 35002).sum().item()
                solver_id = self.solvers[answer_type].match_solver(n_num, n_nums, n_names)
                solver_ids.append(solver_id)

            answer_types = torch.as_tensor(solver_ids, device=self.device)
            for i, solver in enumerate(self.solvers):
                batch_mask = (answer_types == i)
                target_features = features[batch_mask, :]
                if target_features.size(0) > 0:
                    batch_idxes = [bi for bi, m in enumerate(batch_mask) if m]
                    target_num = [num_features[bi] for bi in batch_idxes]
                    target_nums = [nums_features[bi] for bi in batch_idxes]
                    fetched.append((False, batch_mask, target_features, batch_idxes,
                                    target_num, target_nums, None))
                else:
                    fetched.append((True, None, None, None, None, None, None))
        else:
            for i, solver in enumerate(self.solvers[:11]):
                batch_mask = (label_answer_type == i)
                target_features = features[batch_mask, :]
                if target_features.size(0) > 0:
                    batch_idxes = [bi for bi, m in enumerate(batch_mask) if m]
                    target_num = [num_features[bi] for bi in batch_idxes]
                    target_nums = [nums_features[bi] for bi in batch_idxes]
                    equation_targets = [batch['equation_targets'][bi] for bi in batch_idxes]
                    fetched.append((False, batch_mask, target_features, batch_idxes,
                                    target_num, target_nums, equation_targets))
                else:
                    fetched.append((True, None, None, None, None, None, None))
        return fetched, answer_types

    def forward(self, batch, features, answer_types, question_mask):
        num_features = self.extract_num(batch, features, question_mask)
        nums_features = self.extract_nums(batch, features, question_mask)

        loss, accuracy = [], []
        solve_outputs = [None] * features.size(0)
        solve_results = {}
        fetched, answer_types = self.fetch_all(batch, answer_types, features, num_features, nums_features)
        for i, solver in enumerate(self.solvers[:11]):
            skip_solver, batch_mask, target_features, batch_idxes, target_num, target_nums, equation_targets = fetched[i]
            if skip_solver:
                continue
            solve_output, solve_loss, solve_accuracy, solve_result = solver(batch, target_features, target_num, target_nums,
                                                                            equation_targets, batch_mask)
            for si, bi in enumerate(batch_idxes):
                solve_outputs[bi] = solve_output[si]
            if solve_loss is not None:
                loss.append(solve_loss)
                accuracy.append(solve_accuracy)
                solve_results.update(solve_result)

        loss = torch.mean(torch.stack(loss)) if loss else None
        accuracy = torch.mean(torch.stack(accuracy)) if accuracy else None
        return (answer_types, solve_outputs), loss, accuracy, solve_results


class TokenFeatureExtractor(base.Module):
    def __init__(self, token_prefix, config):
        super(TokenFeatureExtractor, self).__init__()
        self.token_prefix = token_prefix
        self.attention = base.AttentionLayer(config)

    def forward(self, batch, features, question_masks):
        wrap_inds = batch[f'{self.token_prefix}_wrap_ind']
        attn_masks = batch[f'{self.token_prefix}_attn_mask']
        pos_masks = batch[f'{self.token_prefix}_pos_mask']

        token_features = []
        for i, (wrap_ind, pos_mask, attn_mask) in enumerate(zip(wrap_inds, pos_masks, attn_masks)):
            if wrap_ind.numel():
                x = features[i, wrap_ind, :]

                question_features = features[i, :].expand(x.size(0), -1, -1)
                question_mask = question_masks[i].expand(x.size(0), -1)

                x = torch.cat((x, question_features), 1)
                attn_mask = torch.cat((attn_mask, question_mask), 1)
                pos_mask = torch.cat((pos_mask, torch.zeros_like(question_mask)), 1)

                x = self.attention(x, attn_mask)
                x = x[pos_mask == 1, :]
                token_features.append(x)
            else:
                token_features.append(None)
        return token_features


class _Equation(base.Module):
    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        raise NotImplementedError()

    def output(self, equation_outputs, loss=None, accuracy=None):
        if loss is not None and accuracy is not None:
            if isinstance(loss, list):
                loss = torch.mean(torch.stack(loss)) if loss and loss[0] is not None else None
            if isinstance(accuracy, list):
                accuracy = torch.mean(torch.stack(accuracy)) if accuracy and accuracy[0] is not None else accuracy
        return equation_outputs, loss, accuracy, {self._key('loss'): loss, self._key('accuracy'): accuracy}

    def _key(self, key):
        if self.training:
            return f"solver(train)/{type(self).__name__[1:]}_{key}"
        else:
            return f"solver(valid)/{type(self).__name__[1:]}_{key}"

    def match_solver(self, n_num, n_nums, n_names):
        if n_num > 0:
            return 6
        elif n_nums > 0 and n_names == 0:
            return -1  # Sum
        elif n_names > 0:
            return 4
        return -99  # Can't find answer


class _EmptyEquation(_Equation):
    def __init__(self, eq_id):
        super(_EmptyEquation, self).__init__()
        self.eq_id = eq_id

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        return None, None, None


class _NumberMatcher(base.Module):
    def __init__(self, hidden_size, p_drop, n_num):
        super(_NumberMatcher, self).__init__()
        self.num_matchers = nn.ModuleList([base.SingleTokenMatcher(hidden_size, p_drop) for _ in range(n_num)])
        self.n_num = n_num

    def forward(self, features, targets, default=0):
        loss, accuracy = [], []

        # Match Number Token Output
        equation_outputs = []
        for i, iter_nf in enumerate(features):
            if torch.is_tensor(iter_nf):
                iter_nf = itertools.repeat(iter_nf, self.n_num)

            equation_output = []
            for ep, (num_feature, matcher) in enumerate(zip(iter_nf, self.num_matchers)):
                if num_feature is None or num_feature.numel() == 0 or num_feature.size(0) == 1:
                    _output = torch.tensor(default, device=self.device)
                else:
                    target = None if targets is None else targets[i][ep]
                    _output, _loss, _accuracy = matcher(num_feature, target)

                    loss.append(_loss)
                    accuracy.append(_accuracy)
                equation_output.append(_output)

            equation_outputs.append(torch.stack(equation_output))

        equation_outputs = torch.stack(equation_outputs)
        if targets:
            return equation_outputs, loss, accuracy
        else:
            return equation_outputs, None, None


class _DiffPerm(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_DiffPerm, self).__init__()
        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 2)
        self.match_type = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        # Match Number Token Output
        equation_outputs, loss, accuracy = self.num_matcher(zip(num_features, nums_features), targets)

        # Match Equation Subtype
        label_2 = None if targets is None else torch.stack([t[2].squeeze() for t in targets])

        x, loss_2, accuracy_2 = self.match_type(features, label_2)
        x = x.argmax(-1)

        equation_outputs = torch.cat((equation_outputs, x.unsqueeze(-1)), -1)

        loss = (torch.mean(torch.stack(loss)) * 2 + loss_2) / 3 if loss else loss_2
        accuracy = (torch.mean(torch.stack(accuracy)) * 2 + accuracy_2) / 3 if accuracy else accuracy_2

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 1 and n_nums >= 1:
            return 0
        return super().match_solver(n_num, n_nums, n_names)


class _CountFromRange(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_CountFromRange, self).__init__()

        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 3)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        num_features = [([None, nf, nf] if nf is None or nf.numel() == 0 or nf.size(0) == 2 else nf) for nf in num_features]
        equation_outputs, loss, accuracy = self.num_matcher(num_features, targets, default=-1)
        if targets is None:
            return self.output(equation_outputs)

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 2:
            return 1
        return super().match_solver(n_num, n_nums, n_names)


class _FindSumFromRange(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_FindSumFromRange, self).__init__()

        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 4)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        num_features = [([None, nf, nf, nf] if nf is None or nf.numel() == 0 or nf.size(0) == 3 else nf) for nf in num_features]
        equation_outputs, loss, accuracy = self.num_matcher(num_features, targets, default=-1)

        if targets is None:
            return self.output(equation_outputs)
        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 3:
            return 2
        return super().match_solver(n_num, n_nums, n_names)


class _WrongMultiply(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_WrongMultiply, self).__init__()
        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 6)
        self.type_matcher = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = self.num_matcher(num_features, targets)

        label_type = None if targets is None else torch.stack([t[6].squeeze() for t in targets])
        x, loss_type, accuracy_type = self.type_matcher(features, label_type)
        x = x.argmax(-1)

        equation_outputs = torch.cat((equation_outputs, x.unsqueeze(-1)), -1)

        loss = (torch.mean(torch.stack(loss)) + loss_type) if loss else loss_type
        accuracy = (torch.mean(torch.stack(accuracy)) * 2 + accuracy_type) / 3 if accuracy else accuracy_type

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 3:
            return 3
        return super().match_solver(n_num, n_nums, n_names)


class _OrderByCompare(_Equation):
    def __init__(self, hidden_size, p_drop, config):
        super(_OrderByCompare, self).__init__()
        self.attention = base.AttentionLayer(config)
        self.binary_classifier = base.BinaryTagging(hidden_size, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        attention_mask = batch['attention_mask'][batch_mask]

        x = self.attention(features, attention_mask)
        if targets is not None:
            targets = torch.stack(targets)
        outputs, loss, accuracy = self.binary_classifier(x, targets, attention_mask)

        input_ids = batch['input_ids'][batch_mask]
        if accuracy is not None:
            output_idx = outputs.argmax(-1, keepdim=True)
            output_id = input_ids.gather(1, output_idx)
            target_idx = targets.argmax(-1, keepdim=True)
            accuracy = torch.mean((output_id == input_ids.gather(1, target_idx)).float())
        return self.output(outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_names >= 1:
            return 4
        return super().match_solver(n_num, n_nums, n_names)


class _HalfSub(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_HalfSub, self).__init__()
        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 2)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = self.num_matcher(num_features, targets)
        if targets is None:
            return self.output(equation_outputs)

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 2:
            return 5
        return super().match_solver(n_num, n_nums, n_names)


class _SumNumSig(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_SumNumSig, self).__init__()
        self.num_classifier = base.SequenceTagging(hidden_size, 3, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = [], [], []
        _target = None
        for i, num_feature in enumerate(num_features):
            if targets is not None:
                _target = targets[i]
            _output, _loss, _accuracy = self.num_classifier(num_feature, _target, None)

            equation_outputs.append(_output)
            loss.append(_loss)
            accuracy.append(_accuracy)

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 1:
            return 6
        return -1


class _MaxSubMin(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_MaxSubMin, self).__init__()
        self.nums_matcher = _NumberMatcher(hidden_size, p_drop, 1)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = self.nums_matcher(nums_features, targets)
        if targets is None:
            return self.output(equation_outputs)

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_nums >= 1:
            return 7
        elif n_num >= 1:
            return 8
        return super(_MaxSubMin, self).match_solver(n_num, n_nums, n_names)


class _MaxSubMin2(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_MaxSubMin2, self).__init__()
        self.choose_num = base.BinaryTagging(hidden_size, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = [], [], []
        _target = None
        for i, num_feature in enumerate(num_features):
            if targets is not None:
                _target = targets[i]
            _output, _loss, _accuracy = self.choose_num(num_feature, _target, None)

            equation_outputs.append(_output)
            loss.append(_loss)
            accuracy.append(_accuracy)

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 1:
            return 8
        elif n_nums >= 1:
            return 7
        return super().match_solver(n_num, n_nums, n_names)


class _CountFromComparePivot(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_CountFromComparePivot, self).__init__()
        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 2)
        self.match_type = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        if targets:
            label_type, label_n = [], []
            for t in targets:
                label_type.append(t[0].squeeze())
                label_n.append(t[1:])
            label_type = torch.stack(label_type)
        else:
            label_type, label_n = None, None

        equation_outputs, loss, accuracy = self.num_matcher(zip(num_features, nums_features), label_n)

        # Match Equation Subtype
        x, loss_type, accuracy_type = self.match_type(features, label_type)
        x = x.argmax(-1)

        equation_outputs = torch.cat((x.unsqueeze(-1), equation_outputs), -1)

        loss = (torch.mean(torch.stack(loss)) * 2 + loss_type) / 3 if loss else loss_type
        accuracy = (torch.mean(torch.stack(accuracy)) * 2 + accuracy_type) / 3 if accuracy else accuracy_type

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num >= 1 and n_nums >= 1:
            return 9
        elif n_num > 3:
            return 10
        return super().match_solver(n_num, n_nums, n_names)


class _CountFromComparePivot2(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_CountFromComparePivot2, self).__init__()
        self.match_num = _NumberMatcher(hidden_size, p_drop, 1)
        self.choose_num = base.BinaryTagging(hidden_size, p_drop)
        self.match_type = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        if targets is not None:
            label_type = torch.stack([t[0].squeeze() for t in targets])
            label_pivot = [[t[1]] for t in targets]
        else:
            label_type, label_pivot = None, None

        # Match Equation Subtype
        x, loss_type, accuracy_type = self.match_type(features, label_type)
        output_type = x.argmax(-1)

        # Match Number
        output_num, loss_num, accuracy_num = self.match_num(num_features, label_pivot)

        # Tag Numbers
        output_tags, loss_tags, accuracy_tags = [], [], []
        label_n = None
        for i, num_feature in enumerate(num_features):
            if targets is not None:
                label_n = targets[i][2]
            _output, _loss, _accuracy = self.choose_num(num_feature, label_n, None)

            output_tags.append(_output)
            if _loss:
                loss_tags.append(_loss)
            if _accuracy:
                accuracy_tags.append(_accuracy)

        loss_tags = torch.mean(torch.stack(loss_tags)) if loss_tags else None
        accuracy_tags = torch.mean(torch.stack(accuracy_tags)) if accuracy_tags else None

        equation_outputs = list(zip(output_type, output_num, output_tags))
        if targets is None:
            return self.output(equation_outputs)
        loss_type = loss_type if loss_type is not None else 0
        loss_num = torch.mean(torch.stack(loss_num)) if loss_num is not None else 0
        loss = (loss_tags * 2 + loss_type + loss_num) / 4 if loss_tags else (loss_type + loss_num) / 2
        if loss == 0 and isinstance(loss, int):
            loss = None

        accuracy_type = accuracy_type if accuracy_type is not None else 0
        accuracy_num = torch.mean(torch.stack(accuracy_num)) if accuracy_num is not None else 0
        accuracy = (accuracy_tags + accuracy_type + accuracy_num) / 3 if accuracy_tags else (accuracy_type + accuracy_num) / 2
        if accuracy == 0 and isinstance(accuracy, int):
            accuracy = None

        return self.output(equation_outputs, loss, accuracy)

    def match_solver(self, n_num, n_nums, n_names):
        if n_num > 3:
            return 10
        elif n_num >= 1 and n_nums >= 1:
            return 9
        return super().match_solver(n_num, n_nums, n_names)
