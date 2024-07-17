from typing import Hashable, Dict, Any, Tuple

from tqdm import tqdm

from legal_generation.utils.cache_run import cache_run


class AbstractScorer:
    scores: Dict[Hashable, Any]

    def reset(self):
        raise NotImplementedError

    def example(self, gen: str, src: str, example_id: Hashable | None = None):
        raise NotImplementedError

    def batch_examples(self, examples: Dict[Hashable, Tuple[str, str]], cache_path: str | None = None):
        # run `example` method on a batch of inputs. Each input is indexed with its id and contains
        # a tuple of (gen, src).
        @cache_run
        def batch_run():
            for eid, (gen, src) in tqdm(examples.items(), total=len(examples), desc=str(self.__class__)):
                self.example(gen, src, eid)
            return self.scores

        scores = batch_run(cache_path=cache_path)
        if set(scores) != set(examples):
            self.reset()
            scores = batch_run(cache_path=cache_path, force_update=True)
        self.scores = scores
