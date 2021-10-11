#! /usr/bin/env python3
import random

_hangul_first_syllables = [
        'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
        'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

_hangul_middle_syllables = [
        'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
        'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
        'ㅣ'
        ]

_hangul_last_syllables = [
        '',
        'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ',
        'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ',
        'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]

_korean_numeral_cardinal_digit = [
        '', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
_korean_numeral_cardinal_unit1 = [
        '', '십', '백', '천']
_korean_numeral_cardinal_unit2 = [
        '', '만 ', '억 ', '조 ', '경 ']
_korean_numeral_ordinal_count = [
        '', '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']
_korean_numeral_ordinal_order = [
        '', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
_korean_numeral_ordinal_tens = [
        '', '열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔']


def is_hangul_char(c):
    if type(c) == str:
        c = ord(c)
    return c >= ord('가') and c <= ord('힣')


def split_hangul_char(c):
    if len(c) != 1:
        raise ValueError("Single character required")
    if not is_hangul_char(c):
        raise ValueError("Hangul character required")

    ml = len(_hangul_middle_syllables)
    ll = len(_hangul_last_syllables)

    o = ord(c) - ord('가')

    return (o // (ml * ll), (o // ll) % ml, o % ll)


def has_last_syllable(s):
    if len(s) == 0:
        raise ValueError("Empty string")

    if not is_hangul_char(s[-1]):
        raise ValueError("Need a hangul character")

    s = split_hangul_char(s[-1])

    return s[-1] != 0


def _num2kor(n):
    s = ''

    u1 = 0
    u2 = 0

    while n > 0 and u2 < len(_korean_numeral_cardinal_unit2):
        d = n % 10000
        n = n // 10000

        if d != 0:
            s = _korean_numeral_cardinal_unit2[u2] + s

            u1 = 0
            while d > 0:
                x = d % 10
                d = d // 10

                if x != 0:
                    if u1 == 0:
                        s = _korean_numeral_cardinal_digit[x] + s
                    elif x == 1:
                        s = _korean_numeral_cardinal_unit1[u1] + s
                    else:
                        s = _korean_numeral_cardinal_digit[x] + _korean_numeral_cardinal_unit1[u1] + s

                u1 += 1
        u2 += 1

    if n != 0:
        raise ValueError("Too large")

    return s


# 일, 이
def num2kor(n):
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return '영'

    return _num2kor(n).strip()

# 하나, 둘
def _num2korcnt(x):
    if x == 0:
        return num2kor(0)
    return _korean_numeral_ordinal_tens[x // 10] + _korean_numeral_ordinal_count[x % 10]


# 한 개, 두 개
def _num2korunit(x):
    if x == 0:
        return num2kor(0)
    elif x == 20:
        return '스무'
    else:
        return _korean_numeral_ordinal_tens[x // 10] + _korean_numeral_ordinal_order[x % 10]


# 첫 번째, 두 번째
def _num2korord(x):
    if x == 1:
        return '첫'
    else:
        return _num2korunit(x)


# 하나, 둘
def num2korcnt(n):
    if n < 0:
        raise ValueError("negative")
    return _num2kor((n // 100) * 100) + _num2korcnt(n % 100)


# 한 개, 두 개
def num2korunit(n):
    if n < 0:
        raise ValueError("negative")
    return _num2kor((n // 100) * 100) + _num2korunit(n % 100)


# 첫 번째, 두 번째
def num2korord(n):
    if n < 0:
        raise ValueError("negative")
    return _num2kor((n // 100) * 100) + _num2korord(n % 100)


def append_connection(target, type):
    tbl = {
            # has last symbol / not
            '은': ['은', '는'],
            '는': ['은', '는'],
            '을': ['을', '를'],
            '를': ['을', '를'],
            '이': ['이', '가'],
            '가': ['이', '가'],
            '와': ['과', '와'],
            '과': ['과', '와'],
            # 하나_나_ 둘_이나_
            '이?': ['이', ''],
            # 산_으로_ 바다_로_
            '으?': ['으', ''],
    }

    if type not in tbl:
        return None

    if len(target) == 0:
        return ''
    if is_hangul_char(target[-1]):
        return tbl[type][0 if has_last_syllable(target) else 1]
    elif target[-1] in '0123456789':
        ntbl = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1]
        return tbl[type][ntbl[int(target[-1])]]
    elif target[-1] in 'ABCDEFGHIJKOPQSTUVWXYZ':
        return tbl[type][1]
    elif target[-1] in 'LMNR':
        return tbl[type][0]
    else:
        return tbl[type][random.randint(0, 1)]
