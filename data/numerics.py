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
    _units = ['개', '살', '송이', '알', '자루', '권', '장', '켤레', '병', '대', '척', '다스', '판', '알', '통', '자리', '숫자', '수', '의 자리',
              '킬로미터', 'km', '㎞', '미터', 'm', '센티미터', 'cm', '㎝', '밀리미터', 'mm', '㎜',
              '킬로리터', 'kl', '㎘', '리터', 'l', 'ℓ', '밀리리터', 'ml', '㎖',
              '세제곱미터', 'm3', 'm^3', '㎥', '세제곱센티미터', 'cm3', 'cm^3', '㎤',
              '제곱킬로미터', 'km2', 'km^2', '㎢', '제곱미터', 'm2', 'm^2', '㎡', '제곱센티미터', 'cm2', 'cm^2', '㎠',
              '톤', 't', '킬로그램', 'kg', '㎏', '그램', 'g', '밀리그램', 'mg', '㎎']

    _1 = "(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉)"
    _1_others = "(?:하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉)"
    _over_10_with_1 = "(?:(?:(?:[일이삼사오육칠팔구]?[백천만])+)?\\s?(?:열|스물|서른|마흔|쉰|예순|일흔|여든|아흔))"
    _over_10_only = "(?:(?:(?:[일이삼사오육칠팔구]?[백천만])+)?\\s?(?:열|스무|서른|마흔|쉰|예순|일흔|여든|아흔))"
    _exp_chinese_1 = f"(?:{'|'.join(_chinese_1.keys())})"
    _exp_chinese_10n = f"(?:{'|'.join(_chinese_10n.keys())})"
    _exp_units = "(?:" + '|'.join(_units) + ")"
    _exp_arabic = "(?:-?\\d+(\\.\\d+)?)"
    _exp_kor = f"(?:(?:{_over_10_with_1}\\s*{_1})|(?:{_over_10_only}|{_1}))"
    _exp_kor_units = f"{_exp_kor}\\s*{_exp_units}"
    _exp_kor_nonunits = f"{_over_10_with_1}\\s*{_1_others}"
    _exp_chn_over_10 = f"(?:{_exp_chinese_1}?{_exp_chinese_10n}\\s*)+(?:{_exp_chinese_1})?"
    _exp_chn = f"(?:{_exp_chn_over_10})"
    _exp_chn_units = f"(?:{_exp_chn}\\s*{_exp_units})|(?:일의 자리)"
    _exp_total_kor = f"(?:{_exp_kor_units})|(?:{_exp_kor_nonunits})|(?:{_exp_chn_units})|(?:{_exp_chn}\\b)"
    _exp_num = f"\\b(?:(?:{_exp_total_kor})|(?:{_exp_arabic}\\s*(?:{_exp_units}|의\\s*자리)?))"
    _exp_nums = f"(?:{_exp_num})(?:,\\s*(?:{_exp_num}))+"
    _exp_all = f"(?:{_exp_nums})|(?:{_exp_num})"
    _exp_num_token = f"(?:\\[NUM\\])|(?:\\[NUMS\\])"

    def __init__(self, num_token, nums_token):
        self.unit_pattern = re.compile(self._exp_units)
        self.c10n_pattern = re.compile(self._exp_chinese_10n)
        self.kor_pattern = re.compile(self._exp_kor)
        self.chn_pattern = re.compile(self._exp_chn)
        self.arabic_pattern = re.compile(self._exp_arabic)
        self.num_pattern = re.compile(self._exp_num)
        self.nums_pattern = re.compile(self._exp_nums)
        self.all_pattern = re.compile(self._exp_all)

        self.num_token_pattern = re.compile(self._exp_num_token)

        self.num_token = num_token
        self.nums_token = nums_token

    def replace_token(self, s):
        replaced = self.nums_pattern.sub(self.nums_token, s)

        replaced = self.num_pattern.sub(
            lambda m: self.num_token + ('' if isinstance((n := self._numeric_info(m.group())), list) or n[2] is None else n[2]),
            replaced)

        nums = [self._numeric_info(m.group()) for m in self.all_pattern.finditer(s)]
        return replaced, nums

    def replace_batch(self, batch):
        replacements, numerics = [], []
        for q in batch:
            replaced, nums = self.replace_token(q)
            replacements.append(replaced), numerics.append(nums)
        return replacements, numerics

    def _numeric_info(self, num_word):
        if ',' in num_word:
            return [self._numeric_info(n) for n in num_word.split(',')]
        if "일의 자리" in num_word:
            return num_word, 1, "의 자리"
        number = self._get_number(num_word := num_word.strip())
        unit = unit.group() if (unit := self.unit_pattern.search(num_word)) else None
        return num_word, number, unit

    def _get_number(self, s):
        if n := self.arabic_pattern.search(s):
            return float(n) if '.' in (n := n.group()) else int(n)
        elif (n := self.kor_pattern.search(s)) or (n := self.chn_pattern.search(s)):
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
    np = NumericProcessor('[NUM]', '[NUMS]')
    print(np.replace_token("상자에 그릇 오천삼백 마흔다섯 개를 꺼내서 먹었는데, 그 중 백이십칠 개가 남았습니다. 1257 개의 상자에는 몇 개의 그릇이 있을까?"))
    print('---')
    print(np.replace_token("접시에 그릇 17개, 38, 122가 있다. [SEP] 접시에 있는 그릇은 다해서 몇 개인지 구하시오. [SEP] 태형이가 접시에서 그릇 열다섯 개를 꺼냈다."))
    print('---')
    print(np.replace_token("가능한 수는 모두 몇 개인지 구하시오. 1, 3, 5, 22 중에서 3 개의 서로 다른 숫자를 뽑아 세 자리 수를 만들려고 한다."))
    print(np.replace_token("1, 3, 5, 22 중에서 3 개의 서로 다른 숫자를 뽑아 세 자리 수를 만들려고 한다. 가능한 수는 모두 몇 개인지 구하시오."))
    print('---')
    print("69부터 83까지 자연수를 쓰려고 합니다. 바르게 계산한 계산한 값은 87210이 될 때, 두 개의 세 자리 수 중 작은 수를 구하시오."
          " 세 자리 수끼리의 곱셈에서 곱해지는 수의 십의 자리 숫자 3을 4로 잘못 보고 계산한 값이 93670이 되었습니다.")
    print(np.replace_token("69부터 83까지 자연수를 쓰려고 합니다. 바르게 계산한 계산한 값은 87210이 될 때, 두 개의 세 자리 수 중 작은 수를 구하시오."
                           " 세 자리 수끼리의 곱셈에서 곱해지는 수의 십의 자리 숫자 3을 4로 잘못 보고 계산한 값이 93670이 되었습니다."))

    print(np.replace_token("10 이상 71 이하의 자연수를 모두 적으려고 합니다. 접시에 7, 3, 0이 적힌 카드가 들어있습니다."
                           " 접시에서 카드를 꺼내서 만들 수 있는 가장 작은 두 자리 수는 얼마입니까?"))
    print(np.replace_token("직사각형의 세로 길이가 408밀리미터일 때, 가로 길이는 몇 mm인지 구하시오. 4, 0, 6, 3 중 1개의 서로 다른 숫자를 뽑아서 한 자리 수를 만들려고 한다."
                           "길이가 186mm인 전선으로 직사각형을 만들었더니, 전선이 딱 맞아 떨어졌다."))
