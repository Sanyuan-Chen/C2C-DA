# coding: utf-8
import random
from typing import *


class RealizerBase:
    def __init__(self, opt, values, *args, **kwargs):
        self.opt = opt
        self.values = values

    def do_realize(self, raw_sentences, *args, **kwargs) -> List[Tuple[list, list]]:
        """
        :param raw_sentences: a list of string, which are generated sentences.
        :param args:
        :param kwargs:
        :return: List of [tokens, labels]
        """
        raise NotImplementedError

    def get_bio_labels(self, value_len, label):
        label = label.replace('<', '').replace('>', '')
        ret = ['B-' + label]
        for tk in range(value_len - 1):
            ret.append('I-' + label)
        return ret


class ThesaurusRealizer(RealizerBase):
    def __init__(self, opt, values,*args, **kwargs):
        super(ThesaurusRealizer, self).__init__(opt, values, *args, **kwargs)

    def do_realize(self, raw_sentences, *args, **kwargs) -> List[Tuple[list, list]]:
        """
        :param raw_sentences: a list of string, which are generated sentences.
        :param args:
        :param kwargs:
        :return: List of [tokens, labels]
        """
        return self.thesaurus_refill(raw_sentences)

    def refill_one_sent(self, sent):
        refilled_sentence = []
        labels = []
        tokens = sent.split()
        for tk in tokens:
            if tk[0] == '<' and tk in self.values:
                value_tokens = random.choice(self.values[tk]).split()
                refilled_sentence.extend(value_tokens)
                labels.extend(self.get_bio_labels(len(value_tokens), tk))
            else:
                refilled_sentence.append(tk)
                labels.append('O')
        if len(refilled_sentence) != len(labels):
            raise ValueError('label and sent length not matched!')
        return refilled_sentence, labels

    def thesaurus_refill(self, raw_sentences):
        refilled_sentences = []
        for t in range(self.opt.thesaurus_times):
            for sent in raw_sentences:
                refilled_sentences.append(self.refill_one_sent(sent))
        return refilled_sentences
