from typing import Hashable
from rouge import Rouge
from nltk import PorterStemmer

from .abstract_scorer import AbstractScorer


class RougeScorer(AbstractScorer):
    def __init__(self):
        self.scores = dict()
        self.stemmer = PorterStemmer()
        self.rouge_eval = Rouge()

    def reset(self):
        self.scores.clear()

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = ' '.join(map(self.stemmer.stem, text.split()))
        if text.strip('.') == '':
            return 'placeholder'
        return text

    def example(self, gen: str, src: str, example_id: Hashable | None = None):
        gen, src = map(self.clean_text, (gen, src))
        try:
            score = self.rouge_eval.get_scores([gen], [src])[0]
        except ValueError:
            score = self.rouge_eval.get_scores(['placeholder'], [src])[0]
        self.scores[example_id or len(self.scores)] = score

    def average_scores(self):
        ret = dict()
        scores = list(self.scores.values())
        for k1 in scores[0]:
            if k1 not in ret:
                ret[k1] = dict()
            for k2 in 'rpf':
                score_list = [s[k1][k2] for s in scores]
                ret[k1][k2] = round(sum(score_list) / len(score_list) * 100, 2)
        return ret
