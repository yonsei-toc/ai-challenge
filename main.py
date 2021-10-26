import fire
from transformers import ElectraModel, ElectraTokenizerFast
from pytorch_lightning import Trainer

from models.agc import AGCModel
from data.datamodule import AGCDataModule, AGCPredictionDataModule


def init_tokenizer(model_name):
    tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[NUM]", "[NUMS]", *[f"[NAME{i}]" for i in range(1, 15)]]})
    return tokenizer


def download(language_model=None):
    print(f"download() : {language_model=}")
    if language_model:
        tokenizer = init_tokenizer(language_model)
        model = ElectraModel.from_pretrained(language_model)
        model.resize_token_embeddings(len(tokenizer.vocab))

        path = f".language-models/{language_model.split('/')[-1]}"

        tokenizer.save_pretrained(path)
        model.save_pretrained(path)


def train(epoch=40, gpu=0, resume=None,
          batch_size=32, augments=200,
          language_model=".language-models/koelectra-base-v3-discriminator", **model_kwargs):
    print(f"train() : {epoch=} {gpu=} {language_model=} {resume=}")
    tokenizer = init_tokenizer(language_model)

    model = AGCModel(language_model, tokenizer, **model_kwargs)

    datamodule = AGCDataModule(tokenizer, batch_size=batch_size, n_aug_per_question=augments)

    trainer = Trainer(max_epochs=epoch, gpus=[gpu], resume_from_checkpoint=resume, stochastic_weight_avg=True)
    trainer.fit(model, datamodule=datamodule)


def sample():
    tokenizer = init_tokenizer(".language-models/koelectra-base-v3-discriminator")
    d = AGCDataModule(tokenizer, batch_size=1, n_aug_per_question=200)
    d.setup('fit')
    dataloader = d.train_dataloader()
    import json

    rows = [row['origin_question'][0] for row in dataloader]
    print(f'Problem generated : {len(rows)}')
    with open("prob.json", 'w', encoding='utf-8-sig') as f:
        json.dump(rows, f, ensure_ascii=False)


def infer():
    lm_path = ".language-models/koelectra-base-v3-discriminator"
    model_path = "model.ckpt"
    data_path = "input.json"

    tokenizer = init_tokenizer(lm_path)
    model = AGCModel.load_from_checkpoint(model_path, tokenizer=tokenizer)
    datamodule = AGCPredictionDataModule(data_path, tokenizer, batch_size=1)
    trainer = Trainer(resume_from_checkpoint=model_path)
    results = trainer.predict(model, datamodule=datamodule, return_predictions=True)
    print(results)


def main(command=None, *_, **__):
    if command:
        return fire.Fire()
    else:
        infer()


if __name__ == '__main__':
    fire.Fire(main)
