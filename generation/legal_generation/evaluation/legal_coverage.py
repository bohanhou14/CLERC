from typing import Dict, Any, Set

from eyecite import get_citations, clean_text, resolve_citations
from eyecite.models import FullCaseCitation, CaseCitation


class LegalCoverage:
    def __init__(self):
        # number of citations in the references (equivalent to #references)
        self.ref_cites = []
        # number of citations that are generated
        self.gen_cites = []
        # number of generated citations that ARE references
        self.true_positive_ref = []
        # number of generated citations that in the references or previous text
        self.true_positive_context = []

    @staticmethod
    def extract_cites(text: str) -> Set[str]:
        if text == '':
            return set()
        text = clean_text(text, ['html', 'all_whitespace']).strip()
        if text == '':
            return set()
        citations = resolve_citations(list(get_citations(text)))
        cases = [
            res for res in citations
            if (isinstance(res.citation, FullCaseCitation) or isinstance(res.citation, CaseCitation))
        ]
        case_labels = [c.citation.token.data for c in cases]
        return set(case_labels)

    def process_text(self, gen: str, ex: Dict[str, Any]):
        gen_cites = self.extract_cites(gen)
        ref_cites = self.extract_cites('\n'.join([c[0] for c in ex['citations']]))
        context_cites = self.extract_cites('\n'.join(ex['short_citations'] + [ex['previous_text']])) | ref_cites

        # saved for use in example generation
        ex['hit_cites'] = list(gen_cites & context_cites - ref_cites)
        ex['wrong_cites'] = list(gen_cites - context_cites)

        self.gen_cites.append(len(gen_cites))
        self.ref_cites.append(len(ref_cites))
        self.true_positive_ref.append(len(gen_cites & ref_cites))
        self.true_positive_context.append(len(gen_cites & context_cites))

    def average_scores(self):
        # micro-average
        ret = {
            'precision_context': sum(self.true_positive_context) / (sum(self.gen_cites) + 1e-3),
            'precision_reference': sum(self.true_positive_ref) / (sum(self.gen_cites) + 1e-3),
            'recall': sum(self.true_positive_ref) / (sum(self.ref_cites) + 1e-3),
            'true_positive_context': sum(self.true_positive_context),
            'generated_citations': sum(self.gen_cites),
            'reference_citations': sum(self.ref_cites),
        }
        return {k: round(v*100, 2) for k, v in ret.items()}
