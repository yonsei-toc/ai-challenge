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
            # _WrongMultiply(),
            # _OrderByCompare(),
            # _HalfSub(),
            # _SumArgs()
        ])
        self.extract_num = TokenFeatureExtractor('num', config)
        self.extract_nums = TokenFeatureExtractor('nums', config)
        self.n_solvers = len(self.solvers)

    def forward(self, batch, features, answer_types):
        num_features = self.extract_num(batch, features)
        nums_features = self.extract_nums(batch, features)

        loss, accuracy = [], []
        label_answer_type = batch['equation_type']
        solve_outputs = []
        solve_results = {}
        for i, solver in enumerate(self.solvers):
            if label_answer_type is not None:
                # batch_mask = batch_mask & (label_answer_type == i)
                batch_mask = (label_answer_type == i)
            else:
                batch_mask = (answer_types == i)

            target_features = features[batch_mask, :]
            if target_features.size(0) == 0:
                continue

            target_num = [n for n, m in zip(num_features, batch_mask) if m]
            target_nums = [n for n, m in zip(nums_features, batch_mask) if m]
            equation_targets = [e for e, m in zip(batch['equation_targets'], batch_mask) if m]

            solve_output, solve_loss, solve_accuracy, solve_result = solver(batch, target_features, target_num, target_nums,
                                                                            equation_targets, batch_mask)
            solve_outputs.append(solve_output)
            if solve_loss is not None:
                loss.append(solve_loss)
                accuracy.append(solve_accuracy)
                solve_results.update(solve_result)

        loss = torch.mean(torch.stack(loss)) if loss else None
        accuracy = torch.mean(torch.stack(accuracy)) if accuracy else None
        return solve_outputs, loss, accuracy, solve_results


class TokenFeatureExtractor(base.Module):
    def __init__(self, token_prefix, config):
        super(TokenFeatureExtractor, self).__init__()
        self.token_prefix = token_prefix
        self.attention = base.AttentionLayer(config)

    def forward(self, batch, features):
        wrap_inds = batch[f'{self.token_prefix}_wrap_ind']
        attn_masks = batch[f'{self.token_prefix}_attn_mask']
        pos_masks = batch[f'{self.token_prefix}_pos_mask']

        token_features = []
        for i, (wrap_ind, pos_mask, attn_mask) in enumerate(zip(wrap_inds, pos_masks, attn_masks)):
            if wrap_ind.numel():
                x = features[i, wrap_ind, :]
                x = self.attention(x, attn_mask)
                x = x[pos_mask == 1, :]
                token_features.append(x)
            else:
                token_features.append(None)
        return token_features


class _Equation(base.Module):
    n_num: int
    n_nums: int

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


class _NumberMatcher(_Equation):
    def __init__(self, hidden_size, p_drop, n_num):
        super(_NumberMatcher, self).__init__()
        self.num_matchers = nn.ModuleList([base.SingleTokenMatcher(hidden_size, p_drop) for _ in range(n_num)])
        self.n_num = n_num

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask, default=0):
        loss, accuracy = [], []

        # Match Number Token Output
        equation_outputs = []
        for i, iter_nf in enumerate(num_features):
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
        return equation_outputs, loss, accuracy


class _DiffPerm(_Equation):
    n_num = 1
    n_nums = 1

    def __init__(self, hidden_size, p_drop):
        super(_DiffPerm, self).__init__()
        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 2)
        self.match_type = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        # Match Number Token Output
        equation_outputs, loss, accuracy = self.num_matcher(batch, features, zip(num_features, nums_features), None, targets, batch_mask)

        # Match Equation Subtype
        label_2 = None if targets is None else torch.stack([t[2].squeeze() for t in targets])

        x, loss_2, accuracy_2 = self.match_type(features, label_2)
        x = x.argmax(-1)

        equation_outputs = torch.cat((equation_outputs, x.unsqueeze(-1)), -1)

        loss = (torch.mean(torch.stack(loss)) * 2 + loss_2) / 3 if loss else loss_2
        accuracy = (torch.mean(torch.stack(accuracy)) * 2 + accuracy_2) / 3 if accuracy else accuracy_2

        return self.output(equation_outputs, loss, accuracy)


class _CountFromRange(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_CountFromRange, self).__init__()

        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 3)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        equation_outputs, loss, accuracy = self.num_matcher(batch, features, num_features, nums_features, targets, batch_mask)
        if targets is None:
            return self.output(equation_outputs)

        return self.output(equation_outputs, loss, accuracy)


class _FindSumFromRange(_Equation):
    def __init__(self, hidden_size, p_drop):
        super(_FindSumFromRange, self).__init__()

        self.num_matcher = _NumberMatcher(hidden_size, p_drop, 4)

    def forward(self, batch, features, num_features, nums_features, targets, batch_mask):
        num_features = [([None, nf, nf, nf] if nf is None or nf.numel() == 0 or nf.size(0) == 3 else nf) for nf in num_features]
        equation_outputs, loss, accuracy = self.num_matcher(batch, features, num_features, nums_features, targets, batch_mask, default=-1)

        if targets is None:
            return self.output(equation_outputs)
        return self.output(equation_outputs, loss, accuracy)


class _WrongMultiply(_Equation):
    def __init__(self):
        super(_WrongMultiply, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _OrderByCompare(_Equation):
    def __init__(self):
        super(_OrderByCompare, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _HalfSub(_Equation):
    def __init__(self):
        super(_HalfSub, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _SumArgs(_Equation):
    def __init__(self):
        super(_SumArgs, self).__init__()

    def forward(self, batch, features, mask):
        pass
