from typing import Hashable, Tuple

import torch
import numpy as np

from .abstract_scorer import AbstractScorer
from .bart_score_copy import BARTScorer


class BartScorer(AbstractScorer):
    def __init__(self, bart_path: str | None, device: str):
        self.bart_path, self.device = bart_path, device
        self.scores = dict()
        self.model: BARTScorer | None = None

    def reset(self):
        self.scores.clear()

    def example(self, gen: str, src: str, example_id: Hashable | None = None):
        if self.model is None:
            self.model = BARTScorer(device=self.device)
            if self.bart_path is not None:
                self.model.load(self.bart_path)

        with torch.no_grad():
            self.scores[example_id] = {
                'p': float(self.model.score([src], [gen], batch_size=1)[0]),
                'r': float(self.model.score([gen], [src], batch_size=1)[0]),
            }

    def get_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        precision, recall = [], []
        for metric in self.scores.values():
            precision.append(metric['p'])
            recall.append(metric['r'])
        precision, recall = np.array(precision), np.array(recall)
        f = (precision + recall) / 2
        return precision, recall, f

    def average_scores(self):
        precision, recall, f = self.get_metrics()
        return {
            'precision': round(float(precision.mean()), 2),
            'recall': round(float(recall.mean()), 2),
            'f': round(float(f.mean()), 2),
        }
