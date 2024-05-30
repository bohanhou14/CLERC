import argparse
import pandas as pd
from tqdm import trange
import tqdm
from data_utils import extract_citations, sread_tsv
from matplotlib import pyplot as plt
import os
import ast
from collections import defaultdict
import time
import random

random.seed(123456)


def build_generation_data(args):
    print(f"Reading full cases from {args.data_folder}...")
    start_time = time.time()
    data = pd.read_csv(os.path.join(args.data_folder, "collection", "collection.majority_only=True.tsv"), sep='\t', names=["docid", "text", "citation"])
    print(f"Read {len(data)} cases in {time.time() - start_time} seconds")

    print(f"Building citation mapping...")
    start_time = time.time()
    citation2docid = defaultdict(list)
    data["citation"] = data.citation.apply(lambda x: ast.literal_eval(x))
    for i, row in tqdm.tqdm(data.iterrows()):
        for citation in row['citation']:
            citation2docid[citation["cite"]].append(row['docid'])
    print(f"Built citation mapping in {time.time() - start_time} seconds")

    # shuffle the dataset and keep the first args.n_cases
    using_data = data.sample(frac=1).reset_index(drop=True).head(args.n_cases)

    print(f"Processing data...")
    start_time = time.time()
    zero_count = 0
    final_data = []
    for i in trange(len(using_data), desc='Processing data'):
        case = using_data.iloc[i]
        text = case['text']
        total_len = len(text)
        total_word_len = len(text.split(" "))
        paragraphs = text.split('\n')
        paragraphs_starts = {p: text.find(p) for p in paragraphs}
        last_third_start = (total_len // 3) * 2 

        # Keep all paragraphs that start after the last third
        paragraphs = [p for p in paragraphs if paragraphs_starts[p] > last_third_start]

        # skip the last two
        paragraphs = paragraphs[:-2]

        # then for every paragraph get the citations
        paragraph_details = []
        for paragraph in paragraphs:
            try:
                case_labels, case_pos = extract_citations(paragraph)
            except:
                case_labels, case_pos = [], []
            paragraph_details.append((paragraph, case_labels, case_pos))

        # randomly select ones with 2+ citations 
        ones_with_2_plus = [p for p in paragraph_details if len(p[1]) >= 2]
        if len(ones_with_2_plus) == 0:
            zero_count += 1
            # print(f"Skipping {case['docid']} as no paragraphs with 2+ citations")
            continue
                
        # store them and all previous text
        potential = []
        for p in ones_with_2_plus:
            # get the index of the paragraph
            idx = paragraphs.index(p[0])
            # get all the previous text
            previous_text = text.split(p[0])[0]

            # get the citations full text
            citations = [citation2docid[c] for c in p[1]]
            # get the full text of the citations
            citations_text = []
            for c_idx, c in enumerate(citations):
                label = p[1][c_idx]
                for docid in c:
                    cur_df = data[data.docid == docid]
                    assert len(cur_df) == 1, f"Expected 1 but got {len(cur_df)} for docid {docid}"
                    citations_text.append((label, cur_df.iloc[0].to_dict()['text']))

            if len(citations_text) < 2:
                continue
                
            # get the full text of the paragraph
            paragraph_text = p[0]
            paragraph_start = text.find(paragraph_text)

            potential.append({
                "docid": f"{case['docid']}-{paragraph_start}",
                "previous_text": previous_text,
                "gold_text": paragraph_text,
                "citations": citations_text
            })
    
        # keep only args.n_paragraphs
        if len(potential):
            chosen_with_two_plus = random.sample(potential, min(args.n_paragraphs, len(potential)))
            final_data.extend(chosen_with_two_plus)

    print(f"Processed data in {time.time() - start_time} seconds")

    # save it out to a new file
    df = pd.DataFrame(final_data)
    print(f"Total length of the data is {len(final_data)} with an average of {len(final_data) / args.n_cases} paragraphs per case.")
    print(f"The number of citations per paragraph is {df.citations.apply(len).mean()} with a max of {df.citations.apply(len).max()} and a min of {df.citations.apply(len).min()}")
    print(f"Skipped {zero_count} cases")
    df.to_json(os.path.join(args.data_folder, "generation", "all.jsonl"), orient='records', lines=True)
        
     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=str, help="Path to the data folder")
    parser.add_argument("--n_cases", type=int, default=105000, help="Number of cases to use")
    parser.add_argument("--n_paragraphs", type=int, default=1, help="Number of paragraphs to use **per case**")
    args = parser.parse_args()
    build_generation_data(args)


    # example usage (100k train, 5k test)
    #   python build_generation_data.py /brtx/archive/orionw/legal/CLERC --n_cases 105000 --n_paragraphs 1





    