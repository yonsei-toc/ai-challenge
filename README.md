# ai-challenge
to be closed after 10.29

# 설치
```shell
pip install -r requirements.txt
```

# 모델 다운
```shell
python main.py download --language-model="monologg/koelectra-base-v3-discriminator"
```

# 학습
```shell
python main.py train --epoch=1 \
                     --language-model=".language-models/koelectra-base-v3-discriminator"
```
- `--epoch` : epoch
- `--language-model` : language model path

## build dictionary
You must build dictionary at first or after it changes.
```shell
script/run dictionary.py
```
## generate dataset
```shell
script/run gen_example.py
```
You should place your copy of `script/gen_example.py` either in `script` directory or in the root directory. if you place it in the root directory, you can execute it directly.
