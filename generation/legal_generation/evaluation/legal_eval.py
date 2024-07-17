from typing import Dict
import os
import json
from dataclasses import dataclass

from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from legal_generation.utils.cache_run import cache_run
from .legal_examples import LegalExample
from .legal_coverage import LegalCoverage
from .rouge_score import RougeScorer
from .bart_scorer import BartScorer


@dataclass
class LegalEval:
    root: str
    bart_path: str
    device: str
    n_example: int

    def __post_init__(self):
        self.did2ex = dict()
        self.generations: Dict[str, str] = dict()
        self.legal_coverage: LegalCoverage = LegalCoverage()
        self.rouge_scorer = RougeScorer()
        self.bart_scorer = BartScorer(self.bart_path, self.device)
        self.legal_examples = LegalExample(self.root)

    def load_data(self):
        clerc = load_dataset('jhu-clsp/CLERC', data_dir='generation', split='test')
        for ins in clerc:
            self.did2ex[ins['docid']] = ins

    def run(self):
        metrics = dict()

        # collect data
        self.load_data()
        for line in self.read_predictions(cache_path=os.path.join(self.root, 'preds.jsonl')):
            gen = line['gen']
            if '<answer>' in gen:
                gen = gen[gen.index('<answer>'):]
            gen = gen.replace('<answer>', '').replace('</answer>', '').strip()
            self.generations[line['meta']['docid']] = gen
        batch_examples = dict()
        for did, ex in self.did2ex.items():
            if did in self.generations:
                batch_examples[did] = (self.generations[did], ex['gold_text'])

        # rouge and bart
        self.rouge_scorer.batch_examples(batch_examples, os.path.join(self.root, 'rouge.json'))
        metrics['rouge'] = self.rouge_scorer.average_scores()
        self.bart_scorer.batch_examples(batch_examples, os.path.join(self.root, 'bartscore.json'))
        metrics['bartscore'] = self.bart_scorer.average_scores()

        # coverage
        for did, ex in tqdm(self.did2ex.items(), total=len(self.did2ex), desc='coverage'):
            if did in self.generations:
                self.legal_coverage.process_text(self.generations[did], ex)
        metrics['citations'] = self.legal_coverage.average_scores()

        with open(os.path.join(self.root, 'metrics.json'), 'w') as fp:
            json.dump(metrics, fp, indent=4)

        # examples
        f = self.bart_scorer.get_metrics()[2]
        tops = np.argsort(-f).tolist()[:self.n_example]
        did_list = list(self.did2ex)
        for top_idx in tops:
            did = did_list[top_idx]
            self.legal_examples.add_example(did, self.generations[did], self.did2ex[did])
        self.legal_examples.index_page()

    @cache_run
    def read_predictions(self):
        # read individual predictions and organize them into jsonl file.
        pred_path = os.path.join(self.root, 'preds')
        lines = []
        for fn in os.listdir(pred_path):
            if fn.endswith('.json'):
                with open(os.path.join(pred_path, fn)) as fp:
                    lines.append(json.load(fp))
        return lines
