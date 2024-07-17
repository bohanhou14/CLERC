# Legal Case Generation

## Environment Setup

This codebase requires Python >= 3.8. 
One should run the fine-tuning on machines with CUDA devices for the best performance.
Our experiments were done with 8 A100 GPU cards.

Please follow the instructions on [pytorch.org](https://pytorch.org) to install the pytorch.
The codes were tested on torch==2.3.1.

The rest of the requirements are listed on requirements.txt.
*After installing pytorch*, you may run the following commands:
```shell
pip install -r requirements.txt
```

Our experiments were done with the Meta-Llama-3 model.
You need to [obtain permission from meta](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to run the codebase.

## Training and Prediction

Run the following commands to fine-tune a Meta-Llama-3-8B model:

```shell
python3 run.py train --cache /path/to/cache/path
```

We have default args. You may run `--help` to see other options.

To make predictions with the fine-tuned model, run

```shell
python3 run.py predict --ckpt /path/to/12345.ckpt
```

The predictions will be saved in a folder beside the checkpoint file.

## Evaluation

The evaluation script can be used for both zero-shot and fine-tuned results.
The output file should follow this format:
```json
{
  "gen": "model generated text",
  "meta": {
    "gold_text": "gold answer"
  }
}
```

Suppose the root folder for evaluation is `/eval/path`, then put the json files under the folder `/eval/path/preds`.
Then run the following commands:
```shell
python3 evaluate.py /pred/path --device cuda:0 --bart-path /path/to/bart
```
where `/path/to/bart` is the path for BARTScorer (see [this page](https://github.com/neulab/BARTScore) for the downloading instructions),
and `device` is the device that BARTScorer will be running on.
