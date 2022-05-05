import abc
import os
import re
from typing import Dict, List, Set, Tuple

import senticnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from senticnet.senticnet import SenticNet


__all__ = ['SenticNet5', 'SenticNet6', 'LMFinance']


class BasicSemtiment(abc.ABC):

    @property
    def negations(self) -> Set[str]:
        return set(['not', 'n\'t', 'less', 'no', 'never', 'nothing', 'nowhere', 
                    'hardly', 'barely', 'scarcely', 'nobody', 'none'])

    @property
    def stopwords(self) -> List[str]:
        return stopwords.words('english')

    @abc.abstractmethod
    def score(self) -> None:
        pass

    def preprocess(self, doc:str) -> List[str]:
        """Preprocess raw string input, including tokenization and lemmatization.

        Args:
            doc: str, document string.

        Returns:
            tokens: list, token size is m sentences * n words.
        """
        if type(doc) is not str:
            raise TypeError("argument doc must be str.")
        if len(doc) == 0:
            return []
        doc   = doc.lower()
        tokens = list()
        sentences = sent_tokenize(doc)
        for sentence in sentences:
            sentence = re.sub(r'[^a-z|\s|\'|0-9]', '', sentence)
            words = word_tokenize(sentence)
            words = [WordNetLemmatizer().lemmatize(w, pos=self.__pos_short(p)) for w, p in pos_tag(words)]
            tokens.append(words)
        return tokens

    def remove_stopwords(self, words: List[str]) -> List[str]:
        return [word for word in words if word not in self.stopwords]

    def __pos_short(self, pos:str) -> str:
        """Convert NLTK POS tags to SWN's POS tags.

        Args:
            pos: str, NLTK pos tags.

        Returns:
            str, SWN's pos tags.
        """
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'a'


class BasicSenticNet(BasicSemtiment, abc.ABC):

    @abc.abstractmethod
    def is_in_lexicon(self) -> None:
        pass

    def combine_phrase_(self, words: List[str]) -> None:
        """

        Args:
            words: list, tokenized sentence.
        """
        i = 0
        while i < len(words):
            phrase, l = self.__check_phrase(words[i:])
            if l > 1:
                words[i] = phrase
                words = words[:i+1] + words[i+l:]
            i += 1

    def __check_phrase(self, words:List[str]) -> Tuple[str, int]:
        """

        Args:
            words: list, tokenized sentence.
        """
        if len(words) == 0:
            return '', 0
        sn = SenticNet()
        phrase, length = words[0], 1
        for i in range(1, min(len(words), self._max_phrase_len)):
            next_phrase = '_'.join(words[:i+1])
            if self.is_in_lexicon(next_phrase):
                phrase, length = next_phrase, i+1
        return phrase, length


class SenticNet5(BasicSenticNet):

    def __init__(self) -> None:
        if 'senticnet6' not in dir(senticnet):
            raise ImportError('your senticnet library is not senticnet 5.')
        self.sentiment_tags = ['polarity', 'pleasantness', 'attention',
                               'sensitivity', 'aptitude']
        self._max_phrase_len = 4

    def is_in_lexicon(self, word: str) -> bool:
        try:
            SenticNet().polarity_intense(word)
        except KeyError:
            return False
        else:
            return True

    def score(self, text:str) -> Dict[str, float]:
        scores = dict(polarity=0, pleasantness=0, attention=0, sensitivity=0, aptitude=0)
        sn = SenticNet()
        sentences = self.preprocess(text)
        for sentence in sentences:
            negative = False
            sentence = self.combine_phrase_(sentence)
            sent_len = max(len(self.remove_stopwords(sentence)), 1)
            for word in sentence:
                if word in self.negations:
                    negative = not negative
                    continue
                if self.is_in_lexicon(word):
                    sentics = sn.sentics(word)
                else:
                    continue
                for sentiment in self.sentiment_tags:
                    if sentiment == 'polarity':
                        score = float(sn.polarity_intense(word)) / sent_len
                    else:
                        score = float(sentics[sentiment]) / sent_len
                    scores[sentiment] -= score if negative else -score
        return scores


class SenticNet6(BasicSenticNet):

    def __init__(self) -> None:
        if 'senticnet6' not in dir(senticnet):
            raise ImportError('your senticnet library is not senticnet 6.')
        self.sentiment_tags = ['polarity', 'introspection', 'temper',
                               'attitude','sensitivity']
        self._max_phrase_len = 4

    def is_in_lexicon(self, word: str) -> bool:
        try:
            SenticNet().polarity_value(word)
        except KeyError:
            return False
        else:
            return True

    def score(self, text:str) -> Dict[str, float]:
        scores = dict(polarity=0, introspection=0, temper=0, attitude=0, sensitivity=0)
        sn = SenticNet()
        sentences = self.preprocess(text)
        for sentence in sentences:
            negative = False
            self.combine_phrase_(sentence)
            sent_len = max(len(self.remove_stopwords(sentence)), 1)
            for word in sentence:
                if word in self.negations:
                    negative = not negative
                    continue
                if self.is_in_lexicon(word):
                    sentics = sn.sentics(word)
                else:
                    continue
                for sentiment in self.sentiment_tags:
                    if sentiment == 'polarity':
                        score = float(sn.polarity_value(word)) / sent_len
                    else:
                        score = float(sentics[sentiment]) / sent_len
                    scores[sentiment] -= score if negative else -score
        return scores


class LMFinance(BasicSemtiment):

    def __init__(self, filepath:str) -> None:
        super().__init__()
        self.sentiment_tags = ['positive', 'negative', 'uncertainty', 'litigious',
                               'weakmodal', 'strongmodal', 'constraining']
        self.__build_lexicon(filepath)
        

    def score(self, text:str) -> Dict[str, float]:
        scores = dict(positive=0, negative=0, uncertainty=0, litigious=0,
                      weakmodal=0, strongmodal=0, constraining=0)
        sentences = self.preprocess(text)
        for sentence in sentences:
            negative = False
            sent_len = max(len(self.remove_stopwords(sentence)), 1)
            for word in sentence:
                if word in self.negations:
                    negative = not negative
                    continue
                for sentiment in self.sentiment_tags:
                    if word in self._lexicon[sentiment]:
                        score = 1 / sent_len
                        scores[sentiment] -= score if negative else -score
        return scores

    def __build_lexicon(self, filepath:str) -> None:
        self._lexicon = dict()
        for sentiment in self.sentiment_tags:
            f = open(os.path.join(filepath, sentiment+'.txt'))
            self._lexicon[sentiment] = [w.strip('\n') for w in f.readlines()]
            f.close()
        return self._lexicon