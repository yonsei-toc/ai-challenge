import re
from generator.dictionary import Dictionary


class NamingProcessor:
    _target_tags = {'entity.name', 'entity.subject'}

    def __init__(self, dict_file='script/dict.json'):
        with open(dict_file, 'r', encoding='utf8') as f:
            dictionary, clskey = Dictionary.load(f.read())
        self.tokens = [str(t) for t in dictionary.tokens if (self._target_tags & t.tags) and len(str(t)) > 1]
        self.tokens.sort(key=len, reverse=True)

    def replace_token(self, s):
        i = 1
        names = {}
        for token in self.tokens:
            if token in s:
                names[f"[NAME{i}]"] = token
                s = s.replace(token, f"[NAME{i}]")
                i += 1
        return s, names

    def replace_batch(self, batch):
        replacements, names = [], []
        for q in batch:
            replaced, nums = self.replace_token(q)
            replacements.append(replaced), names.append(nums)
        return replacements, names


if __name__ == '__main__':
    np = NamingProcessor()
    print(np.replace_token("은지는 윤기보다 많다. [NUM]명 중 가장 가장 많은 사람을 구하시오. 태형은 은지보다 많다. 유정은 윤기보다 적고 태형은 유정보다 적다. 석진은 영수보다 크고 영수는 남준보다 크다."))
