import fire
from transformers import ElectraModel, ElectraTokenizerFast
from pytorch_lightning import Trainer

from models.agc import AGCModel
from data.datamodule import AGCDataModule


def init_language(model_name):
    tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[NUM]", "[NUMS]", "[NAME]"]})

    language_model = ElectraModel.from_pretrained(model_name)
    language_model.resize_token_embeddings(len(tokenizer.vocab))
    return tokenizer, language_model


def download(language_model=None):
    print(f"download() : {language_model=}")
    if language_model:
        tokenizer, model = init_language(language_model)

        path = f".language-models/{language_model.split('/')[-1]}"

        tokenizer.save_pretrained(path)
        model.save_pretrained(path)


def train(epoch=4, gpu=0, resume=None,
          max_seq_len=128, batch_size=32, augments=3,
          language_model=".language-models/koelectra-base-v3-discriminator", **model_kwargs):
    print(f"train() : {epoch=} {gpu=} {language_model=} {resume=}")
    tokenizer, language_model = init_language(language_model)

    model = AGCModel(language_model, tokenizer, **model_kwargs)

    datamodule = AGCDataModule(tokenizer, max_seq_len, batch_size=batch_size, n_aug_per_question=augments)

    trainer = Trainer(max_epochs=epoch, gpus=[gpu], resume_from_checkpoint=resume, stochastic_weight_avg=True)
    trainer.fit(model, datamodule=datamodule)


def infer():
    print("infer()")


def main(command=None, *_, **__):
    if command:
        return fire.Fire()
    else:
        infer()


if __name__ == '__main__':
    fire.Fire(main)
