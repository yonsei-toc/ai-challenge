import re


class NumericProcessor:
    _pures = {
        '한': 1, '하나': 1, '두': 2, '둘': 2, '세': 3, '셋': 3, '네': 4, '넷': 4,
        '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9,
        '열': 10, '스물': 20, '스무': 20, '서른': 30, '마흔': 40, '쉰': 50, '예순': 60, '일흔': 70, '여든': 80, '아흔': 90
    }
    _chinese_1 = {
        '일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9
    }
    _chinese_10n = {
        '십': 10, '백': 100, '천': 1000, '만': 10000
    }
    _units = ['개', '살', '송이', '알', '자루', '권', '장', '켤레', '병', '대', '척', '다스', '판', '알', '통',
              '킬로미터', 'km', '미터', 'm', '센티미터', 'cm', '밀리미터', 'mm',
              '킬로리터', 'kl', '리터', 'l', '밀리리터', 'ml',
              '세제곱미터', 'm3', 'm^3', '세제곱센티미터', 'cm3', 'cm^3',
              '제곱킬로미터', 'km2', 'km^2' '제곱미터', 'm2', 'm^2', '제곱센티미터', 'cm2', 'cm^2',
              '톤', 't', '킬로그램', 'kg', '그램', 'g', '밀리그램', 'mg']

    _1 = "(한|두|세|네|다섯|여섯|일곱|여덟|아홉)"
    _1_others = "(하나|둘|셋|넷)"
    _over_10_with_1 = "((([일이삼사오육칠팔구]?[백천만])+)?(\\ )?(열|스물|서른|마흔|쉰|예순|일흔|여든|아흔))"
    _over_10_only = "((([일이삼사오육칠팔구]?[백천만])+)?(\\ )?(열|스무|서른|마흔|쉰|예순|일흔|여든|아흔))"
    _exp_chinse_10n = f"({'|'.join(_chinese_10n.keys())})"
    _exp_units = "(\\ ?" + '|\\ ?'.join(_units) + ")"
    _exp_arabic = "(\\d(\\.\\d+)?)+"
    _exp_kor = f"(({_over_10_with_1}{_1})|({_over_10_only}|{_1}))"
    _exp_all = f"({_exp_kor}{_exp_units})|({_exp_arabic}{_exp_units}?)"

    def __init__(self, num_token):
        self.unit_pattern = re.compile(self._exp_units)
        self.c10n_pattern = re.compile(self._exp_chinse_10n)
        self.kor_pattern = re.compile(self._exp_kor)
        self.arabic_pattern = re.compile(self._exp_arabic)
        self.all_pattern = re.compile(self._exp_all)
        self.num_token = num_token

    def replace_token(self, s):
        nums = [self._numeric_info(m.group()) for m in self.all_pattern.finditer(s)]
        replaced = self.all_pattern.sub(self.num_token, s)
        return replaced, nums

    def replace_batch(self, batch):
        replacements, numerics = [], []
        for q in batch:
            replaced, nums = self.replace_token(q)
            replacements.append(replaced), numerics.append(nums)
        return replacements, numerics

    def _numeric_info(self, num_word):
        number = self._get_number(num_word)
        unit = unit.group() if (unit := self.unit_pattern.search(num_word)) else None
        return num_word, number, unit

    def _get_number(self, s):
        if n := self.arabic_pattern.search(s):
            return float(n) if '.' in (n := n.group()) else int(n)
        elif n := self.kor_pattern.search(s):
            n = n.group()
            result = 0
            # 맨 뒤에 하나, 둘, 스물셋 등 계산
            for k, v in self._pures.items():
                if len(n) > len(n := n.replace(k, '')):
                    result += v

            # 십, 백, 천 등 단위 계산
            mul = 0
            for t in self.c10n_pattern.split(n):
                if len(t := t.strip()) == 0:
                    continue

                if t in self._chinese_1:
                    mul = self._chinese_1[t]
                elif t in self._chinese_10n:
                    result += (mul if mul > 0 else 1) * self._chinese_10n[t]
                    mul = 0
            result += mul
            return result

        raise ValueError(f"Can't get number from {s}")


if __name__ == "__main__":
    np = NumericProcessor('[NUM]')
    print(np.replace_token("상자에 그릇 오천삼백 마흔다섯 개를 꺼내서 먹었는데, 그 중 백이십칠 개가 남았습니다. 1257 개의 상자에는 몇 개의 그릇이 있을까?"))
    print(np.replace_token("접시에 그릇 17개가 있다. [SEP] 접시에 있는 그릇은 다해서 몇 개인지 구하시오. [SEP] 태형이가 접시에서 그릇 열다섯 개를 꺼냈다."))
