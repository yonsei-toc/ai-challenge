from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from data.preprocessor import TrainingPreprocessor


class AGCDataModule(LightningDataModule):
    def __init__(self, tokenizer, max_seq_len, n_aug_per_question=3, split=None, batch_size=32, wrap_numeric=2):
        super(AGCDataModule, self).__init__()
        self.n_aug_per_question = n_aug_per_question
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_preprocessor = TrainingPreprocessor(self.tokenizer, self.max_seq_len, wrap_numeric)
        self.val_preprocessor = TrainingPreprocessor(self.tokenizer, self.max_seq_len, wrap_numeric, injection_prob=0.7)

    def setup(self, stage=None):
        if stage == "fit":
            from generator.build import build_problems
            generate_problem, problem_ids = build_problems()
            targets = problem_ids * self.n_aug_per_question
            self.train_dataset = AGCTrainDataset(targets, generate_problem)
            self.val_dataset = AGCTrainDataset(problem_ids * 10, generate_problem)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_preprocessor, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.val_preprocessor, batch_size=self.batch_size, shuffle=True)


class AGCTrainDataset(Dataset):
    def __init__(self, targets, generate_problem):
        super(AGCTrainDataset, self).__init__()
        self.targets = targets
        self.generate_problem = generate_problem

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        problem_id = self.targets[idx]
        problem = self.generate_problem(problem_id)
        return problem
