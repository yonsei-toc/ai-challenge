from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from data.preprocessor import TrainingPreprocessor


class AGCDataModule(LightningDataModule):
    def __init__(self, tokenizer, max_seq_len, n_aug_per_question=3, split=None, batch_size=32):
        super(AGCDataModule, self).__init__()
        self.n_aug_per_question = n_aug_per_question
        self.batch_size = batch_size
        self.train_dataset = None
        self.test_dataset = None
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_preprocessor = TrainingPreprocessor(self.tokenizer, self.max_seq_len)

    def setup(self, stage=None):
        if stage == "fit":
            from data_generator import init, templates, TokenSelector
            target_idxs = list(range(len(templates.fns))) * self.n_aug_per_question
            dictionary = init()
            token_selector = TokenSelector
            self.train_dataset = AGCTrainDataset(target_idxs, dictionary, token_selector, list(templates.fns))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_preprocessor, batch_size=self.batch_size)


class AGCTrainDataset(Dataset):
    def __init__(self, target_idxs, dictionary, token_selector, fns):
        super(AGCTrainDataset, self).__init__()
        self.target_idxs = target_idxs
        self.dictionary = dictionary
        self.token_selector = token_selector
        self.fns = fns

    def __len__(self):
        return len(self.target_idxs)

    def __getitem__(self, idx):
        fn_idx = self.target_idxs[idx]

        fn = self.fns[fn_idx]
        tokens = self.token_selector(self.dictionary)
        item = fn(tokens)
        return item
