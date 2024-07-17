from typing import *
import os
import json

from transformers import AutoTokenizer
from lightning.pytorch.callbacks import Callback
from lightning import pytorch as pl


class GenerationSaver(Callback):
    def __init__(self, save_path: str, pretrained: str):
        self.save_path = save_path
        self.predicts = []
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for i, out in enumerate(outputs):
            out_text = self.tokenizer.decode(out, skip_special_tokens=True)
            self.predicts.append({'meta': batch['meta'][i], 'gen': out_text})

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.predicts:
            return
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, f'{trainer.global_rank}.jsonl'), 'w') as fp:
            fp.write('\n'.join(map(json.dumps, self.predicts)))
        self.predicts = []
