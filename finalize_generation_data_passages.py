import pandas as pd
import numpy as np
import os
import pickle
from transformers import set_seed, BertTokenizerFast
from tqdm import tqdm
import re
import torch
import argparse

from rankllama import load_rank_llama, rank_batch, rank


def chunk(text: str, max_len: int = 250, overlap=0.5):
    # return a list of chunks of text split into `max_len` chunks with overlap of `overlap`
    if max_len <= 0:
        raise ValueError("max_len must be a positive integer")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")

    words = text.split()
    chunks = []
    step = int(max_len * (1 - overlap))  # Calculate step size based on overlap percentage
    for i in range(0, len(words), step):
        chunk_words = words[i:i+max_len]
        if chunk_words:  # Ensure we don't append empty chunks
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

    return chunks


def finalize_generation_data(args):
    # assert we have a gpu
    assert torch.cuda.is_available(), "No GPU available"

    print(f"Loading data...")
    data = pd.read_json(os.path.join(args.data_folder, "generation", "all.jsonl"), lines=True)
    print(f"Loaded data with {len(data)} instances")
    # remove those whose citations are more than 10
    data = data[data.citations.apply(len) <= 10]
    print(f"Removed those with more than 10 citations, now have {len(data)} instances")

    # sample only args.total_size
    data = data.sample(args.total_size, random_state=123456)
    print(f"Sampled {args.total_size} instances")

    if args.debug:
        data = data.head(100)

    citations = data["citations"].tolist()
    # make a mapping of these
    doc_index = []
    doc_text = []
    queries = []
    doc_titles = []
    for idx, citation_list in enumerate(citations):
        doc_titles.extend([item[0] for item in citation_list])
        doc_index.extend([idx]*len(citation_list))
        doc_text.extend([item[1] for item in citation_list])
        queries.extend([data.iloc[idx]["gold_text"]] * len(citation_list))

    assert len(doc_index) == len(doc_text) == len(queries), f"Lengths don't match: {len(doc_index)}, {len(doc_text)}, {len(queries)}"

    print("Loading model and tokenizer")
    tokenizer, model = load_rank_llama()
   
    print(f"Ranking citations...")
    scores_for_df = []
    batch_size = args.batch_size
    best_chunks = []
    for i, citation_text in enumerate(tqdm(doc_text, total=len(doc_text))):
        title = doc_titles[i]
        passages = [title + "\n\n" + item for item in chunk(citation_text)]
        # batch the passages
        batch_size = 6
        all_scores = []
        query = queries[i]
        for j in range(0, len(passages), batch_size):
            batch = passages[j:j+batch_size]
            scores = rank_batch([query]*len(batch), batch, tokenizer, model)
            all_scores.extend(scores)
        
        # take the top one and append it
        best_passage = passages[np.argmax(all_scores)]
        best_chunks.append(best_passage)
    
    # put them back into list of list format
    new_citations = []
    for citation_idx, _ in enumerate(citations):
        ones_in_cur_index = [chunk for idx, chunk in zip(doc_index, best_chunks) if idx == citation_idx]
        new_citations.append(ones_in_cur_index)

    data["short_citations"] = new_citations

    # split into train and test, via args.test_size
    train = data[:-args.test_size]
    test = data[-args.test_size:]

    # save out the train and test
    train.to_json(os.path.join(args.data_folder, "generation", "train.jsonl"), orient='records', lines=True)
    test.to_json(os.path.join(args.data_folder, "generation", "test.jsonl"), orient='records', lines=True)
    print(f"Saved out train and test to {os.path.join(args.data_folder, 'generation')}")
    print(f"Sizes are train: {len(train)} and test: {len(test)}")    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--total_size", type=int, default=6000, help="Total size of the dataset")
    parser.add_argument("--test_size", type=int, default=1000, help="Size of the test set")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for ranking")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    args = parser.parse_args()
    finalize_generation_data(args)

    #   python finalize_generation_data_passages.py /brtx/archive/orionw/legal/CLERC 

    # load from HF with
    #       dataset = load_dataset("jhu-clsp/CLERC", data_files={"data": f"generation/test.jsonl"})["test"]

