import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.trainer import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins import DeepSpeedPrecision

from legal_generation.modeling.ln_model import ReportGenerationModel
from legal_generation.data_io.clerc import load_data
from legal_generation.modeling.save_callback import GenerationSaver


def get_save_path(args):
    if args.o is not None:
        return args.o
    assert args.ckpt is not None, 'please provide output path'
    name = os.path.basename(args.ckpt)
    if name.endswith('.ckpt'):
        name = name[:-len('.ckpt')]
    return os.path.join(os.path.dirname(args.ckpt), name + '.predict')


def main():
    parser = ArgumentParser()
    parser.add_argument('action', choices=['train', 'test'])
    parser.add_argument('--cache', default=os.path.join(os.environ.get('HOME'), 'fullattn'))
    parser.add_argument('--pretrained', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--warmup', default=2000, type=int)
    parser.add_argument('--strategy', default="deepspeed", type=str, choices=['deepspeed', 'ddp'])
    parser.add_argument('--n-gpu', default=8, type=int)
    parser.add_argument('--precision', default='bf16-mixed', type=str)
    parser.add_argument('--exp', default='debug', type=str)
    parser.add_argument('--acc', default=4, type=int)
    parser.add_argument('--max-length', default=6000, type=int)
    parser.add_argument('--n-val', default=1000, type=int)
    parser.add_argument('--check-interval', default=1000, type=int)
    parser.add_argument('--save-top-k', default=10, type=int)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--use-ref', action=BooleanOptionalAction, default=True)
    parser.add_argument('-o', type=str)
    parser.add_argument('--max-new', type=int, default=512)
    parser.add_argument('--data', default='clerc', type=str)
    args = parser.parse_args()
    if args.action == 'train':
        logger = pl_loggers.TensorBoardLogger(args.cache, args.exp)
        callbacks = [
            ModelCheckpoint(
                os.path.join(logger.log_dir, 'ckpt'), monitor='dev_loss', mode='min', save_top_k=args.save_top_k,
                save_last=True, filename='{step:06d}', auto_insert_metric_name=False,
            ),
            LearningRateMonitor('step'),
        ]
    elif args.action == 'test':
        callbacks = [GenerationSaver(get_save_path(args), args.pretrained)]
        logger = pl_loggers.TensorBoardLogger(os.path.join('/tmp/reportgen', args.exp))
        args.strategy = 'ddp'
    else:
        raise NotImplementedError

    args.n_gpu = min(args.n_gpu, torch.cuda.device_count())
    if args.n_gpu > 0 and torch.cuda.is_available():
        if args.strategy == "deepspeed":
            gpu_kwargs = {
                'plugins': [DeepSpeedPrecision(args.precision)],
                'strategy': "deepspeed_stage_3",
                'devices': args.n_gpu,
            }
        elif args.strategy == 'ddp':
            gpu_kwargs = {
                'precision': args.precision,
                'strategy': DDPStrategy('gpu'),
                'devices': args.n_gpu,
            }
        else:
            raise NotImplementedError
    else:
        gpu_kwargs = {'accelerator': 'cpu'}

    trainer = Trainer(
        log_every_n_steps=20, use_distributed_sampler=gpu_kwargs['devices'] > 1, gradient_clip_val=.8,
        gradient_clip_algorithm='norm', max_epochs=128, logger=logger, enable_progress_bar=True,
        callbacks=callbacks, accumulate_grad_batches=args.acc, check_val_every_n_epoch=1,
        val_check_interval=args.check_interval, **gpu_kwargs,
    )

    if args.ckpt is None:
        model = ReportGenerationModel(
            pretrained=args.pretrained, lr=args.lr, warmup=args.warmup, lora_rank=32,
            max_new=args.max_new
        )
    else:
        model = ReportGenerationModel.load_from_checkpoint(args.ckpt, strict=False)

    train_dl, val_dl = load_data(
        bsz=1, pretrained=args.pretrained, max_length=args.max_length, shuffle=True, use_ref=args.use_ref
    )
    if args.action == 'train':
        model.train()
        trainer.fit(model, train_dl, val_dl)
    elif args.action == 'test':
        trainer.predict(model, val_dl)


if __name__ == '__main__':
    main()
