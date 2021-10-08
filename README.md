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

# 데이터셋 테스트
```shell
cd data_generator
python generator.py
```
