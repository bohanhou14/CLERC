import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import os


def load_model_8bit():
    model = AutoModelForSequenceClassification.from_pretrained(
        '/home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage',
        device_map='auto',
        # load_in_8bit=True,
        torch_dtype=torch.float16,
    ).cuda()
    return model
    

def get_model(peft_model_name):
    if os.path.isdir("/home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage"):
        print(f"Loading saved model from /home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage")
        model = load_model_8bit()
    else:
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.eval()
        # save full model to `my_legal_data/rankllama-v1-7b-lora-passage`
        print(f"Saving model to /home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage")
        model.save_pretrained("/home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage")
        print(f"Reloading model from /home/oweller2/brtx601_1/legal/my_legal_data/rankllama-v1-7b-lora-passage")
        model = load_model_8bit()
    model.config.pad_token_id = model.config.eos_token_id
    model.config.max_length = 512
    return model


def load_rank_llama():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', model_max_length=512)
    model = get_model('castorini/rankllama-v1-7b-lora-passage')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.max_length = 768
    model.max_length = 768
    return tokenizer, model


def rank(query, passage, tokenizer, model):
    inputs = tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt')
    # model = model.cuda()
    with torch.no_grad():
        # put inputs on the GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        score = logits[0][0]
        return score.item()


def rank_batch(queries, passages, tokenizer, model):
    assert len(queries) == len(passages)
    queries = [f'query: {query}' for query in queries]
    passages = [f'document: {passage}' for passage in passages]
    inputs = tokenizer(queries, passages, return_tensors='pt', padding=True, truncation=True)
    # model = model.cuda()
    with torch.no_grad():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        scores = logits[:, 0]
        return scores.tolist()