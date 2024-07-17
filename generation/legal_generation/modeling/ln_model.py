import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch import Tensor
from lightning.pytorch import LightningModule
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class ReportGenerationModel(LightningModule):
    def __init__(self, pretrained: str, lora_rank: int, warmup: int, lr: float, max_new: int = 768):
        super().__init__()
        self.pretrained, self.lora_rank, self.max_new = pretrained, lora_rank, max_new
        self.warmup, self.lr = warmup, lr
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_rank, lora_alpha=32, lora_dropout=0.1,
            target_modules=['q_proj', 'v_proj', 'o_proj'],
        )
        self.lora_model = get_peft_model(AutoModelForCausalLM.from_pretrained(pretrained), peft_config, 'lora_decoder')
        self.save_hyperparameters()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        if isinstance(self.trainer.strategy, DeepSpeedStrategy) and False:
            from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
            if "offload_optimizer" in self.trainer.strategy.config["zero_optimization"].keys():
                optim_class = DeepSpeedCPUAdam
            else:
                optim_class = FusedAdam
        else:
            optim_class = torch.optim.Adam
        optimizer = optim_class(params, lr=self.lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, self.warmup)
        scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], scheduler_config

    def transformer_forward(self, input_ids: Tensor, attention_mask: Tensor, skip: Tensor):
        bsz, seq_len = input_ids.shape
        labels = input_ids.clone()
        label_mask = torch.arange(seq_len, device=labels.device)[None, :].expand(bsz, seq_len) >= skip[:, None]
        labels[~label_mask] = -100
        model_out = self.lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return model_out

    def training_step(self, batch, batch_idx):
        model_out = self.transformer_forward(batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['skip'])
        self.log('loss', model_out.loss, prog_bar=True)
        return model_out.loss

    def validation_step(self, batch, batch_idx):
        model_out = self.transformer_forward(batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['skip'])
        self.log('dev_loss', model_out.loss, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        bsz = batch['tgt_input_ids'].size(0)
        ret = []
        for i in range(bsz):
            input_ids = batch['tgt_input_ids'][i][batch['tgt_attention_mask'][i]][:batch['skip'][i]]
            model_gen = self.lora_model.generate(input_ids=input_ids[None, :], max_new_tokens=self.max_new)
            gen = model_gen[0][len(input_ids):].cpu().tolist()
            ret.append(gen)
        return ret
