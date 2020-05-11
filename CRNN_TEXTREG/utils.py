import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

# ky tu *

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '*'

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):

        length = []
        result = []

        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                r.append(index)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []

        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)

            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length

            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)

        else:
            assert t.numel() == length.sum()
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw
                    )
                )
                index += l
            return texts