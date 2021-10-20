import torch
import torch.nn as nn
import models.base as base


class TemplateSolver(nn.Module):
    def __init__(self, hidden_size, p_drop, config):
        super(TemplateSolver, self).__init__()
        self.solvers = nn.ModuleList([
            _DiffPerm(hidden_size, p_drop),
            _CountFromRange(),
            _FindSumFromRange(),
            _WrongMultiply(),
            _OrderByCompare(),
            _HalfSub(),
            _SumArgs()
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
            solve_output, solve_loss, solve_accuracy = solver(batch, target_features,
                                                              target_num, target_nums, batch_mask)
            solve_outputs.append(solve_output)
            if solve_loss is not None:
                loss.append(solve_loss)
                accuracy.append(solve_accuracy)
            break

        loss = torch.stack(loss)
        accuracy = torch.stack(accuracy)
        return solve_outputs, torch.mean(loss), torch.mean(accuracy)


class TokenFeatureExtractor(nn.Module):
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


class _Equation(nn.Module):
    n_num: int
    n_nums: int

    def forward(self, batch, features, num_features, nums_features, batch_mask):
        raise NotImplementedError()


class _DiffPerm(_Equation):
    n_num = 1
    n_nums = 1

    def __init__(self, hidden_size, p_drop):
        super(_DiffPerm, self).__init__()
        self.match_num = base.SingleTokenMatcher(hidden_size, p_drop)
        self.match_nums = base.SingleTokenMatcher(hidden_size, p_drop)
        self.match_type = base.SequenceClassifier(hidden_size, 4, p_drop)

    def forward(self, batch, features, num_features, nums_features, batch_mask):
        loss, accuracy = [], []

        # Match Number Token Output
        equation_outputs = []
        equation_targets = batch['equation_targets']
        for i, (num_feature, nums_feature) in enumerate(zip(num_features, nums_features)):
            num_0, nums_1 = None, None
            if num_feature is not None and num_feature.size(0) >= self.n_num:
                if equation_targets is None:
                    num_0, _, _ = self.match_num(num_feature, None)
                else:
                    num_0, loss_0, accuracy_0 = self.match_num(num_feature, equation_targets[i][0])
                    loss.append(loss_0)
                    accuracy.append(accuracy_0)

            if nums_feature is not None and nums_feature.size(0) >= self.n_nums:
                if equation_targets is None:
                    nums_1, _, _ = self.match_nums(nums_feature, None)
                else:
                    nums_1, loss_1, accuracy_1 = self.match_nums(nums_feature, equation_targets[i][1])
                    loss.append(loss_1)
                    accuracy.append(accuracy_1)

            if num_0 is not None and nums_1 is not None:
                equation_outputs.append(torch.stack((num_0, nums_1)))
        equation_outputs = torch.stack(equation_outputs)

        # Match Equation Subtype
        if equation_targets is None:
            x, _, _ = self.match_type(features, None)
        else:
            label_2 = torch.stack([e[2] for e in equation_targets])[batch_mask]
            x, loss_2, accuracy_2 = self.match_type(features, label_2)
            loss = torch.mean(torch.stack(loss))
            accuracy = torch.stack(accuracy)

        x = x.argmax(-1)
        equation_outputs = torch.cat((equation_outputs, x.unsqueeze(-1)), -1)

        return equation_outputs, loss, accuracy


class _CountFromRange(_Equation):
    def __init__(self):
        super(_CountFromRange, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _FindSumFromRange(_Equation):
    def __init__(self):
        super(_FindSumFromRange, self).__init__()

    def forward(self, batch, features, mask):
        pass


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
