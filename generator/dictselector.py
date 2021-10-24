from . import dictionary
import random


class DictionarySelector():
    __slots__ = ('_master_dict', '_picked')
    def __init__(self, dictionary):
        self._master_dict = dictionary
        self._picked = set()

    def get(self, key: dictionary.Dictionary.key_t):
        tokens = self._master_dict.resolve_items(key) - self._picked

        if len(tokens) == 0:
            raise RuntimeError("Insufficient tokens; current picked: {}"
                    .format(self._picked))

        ret = random.choice(list(tokens))
        self._picked.add(ret)

        return ret

    def unsafe_get_all(self, key: dictionary.Dictionary.key_t):
        return self._master_dict.resolve_items(key) - self._picked

    def unsafe_get(self, key: dictionary.Dictionary.key_t):
        return random.choice(self.unsafe_get_all(key))
