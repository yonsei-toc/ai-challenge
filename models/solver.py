import torch.nn as nn
import models.base as base


class TemplateSolver(nn.Module):
    def __init__(self, hidden_size, p_drop):
        super(TemplateSolver, self).__init__()
        self.solvers = nn.ModuleList([
            _DiffPerm(),
            _CountFromRange(),
            _FindSumFromRange(),
            _WrongMultiply(),
            _OrderByCompare(),
            _HalfSub(),
            _SumArgs()
        ])
        self.n_solvers = len(self.solvers)

    def forward(self, batch, features, answer_types):
        loss = 0
        for i, solver in enumerate(self.solvers):
            mask = (answer_types == i)
            selected_features = features[mask, :]
            if selected_features.size(0) == 0:
                continue
            solve_outputs, solve_loss, solve_accuracy = solver(batch, selected_features, mask)
            loss += solve_loss
        return None, loss, None


class _DiffPerm(nn.Module):
    def __init__(self):
        super(_DiffPerm, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _CountFromRange(nn.Module):
    def __init__(self):
        super(_CountFromRange, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _FindSumFromRange(nn.Module):
    def __init__(self):
        super(_FindSumFromRange, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _WrongMultiply(nn.Module):
    def __init__(self):
        super(_WrongMultiply, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _OrderByCompare(nn.Module):
    def __init__(self):
        super(_OrderByCompare, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _HalfSub(nn.Module):
    def __init__(self):
        super(_HalfSub, self).__init__()

    def forward(self, batch, features, mask):
        pass


class _SumArgs(nn.Module):
    def __init__(self):
        super(_SumArgs, self).__init__()

    def forward(self, batch, features, mask):
        pass
