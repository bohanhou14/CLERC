from typing import Dict, Any, List

from transformers import PreTrainedTokenizer, AutoTokenizer


class LazyTokenizer:
    pretrained: str
    tokenizer_kwargs: Dict[str, Any] | None = None
    _tokenizer: PreTrainedTokenizer | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained, **(self.tokenizer_kwargs or dict())
            )
        return self._tokenizer
    
    def tok(self, text: str, special=False):
        return self.tokenizer(text, add_special_tokens=special)['input_ids']

    def remove_bos_eos(self, text: str) -> str:
        if text.startswith(self.tokenizer.bos_token):
            text = text[len(self.tokenizer.bos_token):]
        if text.endswith(self.tokenizer.eos_token):
            text = text[:-len(self.tokenizer.eos_token)]
        return text
    
    def add_bos_eos_ids(self, ids: List[int]):
        return [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
