import lzma
import json
import re
from tqdm import trange, tqdm
from nltk.tokenize import sent_tokenize
from CLERC.data.src.data_utils import read_tsv, extract_citations, clean_cstr
from transformers import AutoTokenizer
import pandas as pd
import os
import pickle

def build_doc_collection(data_path = "/srv/local1/bhou4/data.jsonl.xz", majority_only = False):
    with lzma.open(data_path) as f:
        case_ids = []
        case_bodies = []
        citations = []
        lines = f.readlines()
        for line in tqdm(lines, desc='Building collections...', total=len(lines)):
            case = json.loads(str(line, 'utf8'))
            if len(case['casebody']['data']['opinions']) == 0:
                continue
            if majority_only:
                potential_majority_type = case['casebody']['data']['opinions'][0]['type']
                potential_majority_text = case['casebody']['data']['opinions'][0]['text']
                paras = potential_majority_text.split("\n")
                if potential_majority_type == 'majority' and len(paras) > 12:
                    case_ids.append(case['id'])
                    citations.append(case['citations'])
                    case_bodies.append(potential_majority_text) 
            else:
                body = []
                for item in case['casebody']['data']['opinions']:
                    body.append(item['text'])
                case_ids.append(case['id'])
                citations.append(case['citations'])
                case_bodies.append(" ".join(body))
    return case_ids, case_bodies, citations

def build_citations_tsv(data_path="data.jsonl.xz", save_dir=".", majority_only=False):
    case_ids, case_bodies, citations = build_doc_collection(data_path = data_path, majority_only=majority_only)
    # with open(save_dir + "cites2did.pkl", 'rb') as f:
    #     keys = case_ids
    #     values = citations
    #     dic = dict(map(lambda i,j : (i,j) , keys,values))
    #     pickle.dump(dic, f)
    d = {'did': case_ids, 'text': case_bodies, 'citations': citations}
    data = pd.DataFrame(data=d)
    save_path = os.path.join(save_dir, f"collection.majority_only={majority_only}.tsv")
    data.to_csv(save_path, sep='\t', header=False, index=False)
    # with open(save_path, 'w') as f:
    #     for i in trange(len(case_ids), desc='Saving collections...'):
    #         f.write(f"{case_ids[i]}\t{case_bodies[i]}\t{citations[i]}\n")

def build_passages_sw(case_ids, case_bodies, max_len = 350, sw_len=175):
    passages = []
    doc_ids = []
    # total_passages_cites = []
    for i in trange(len(case_bodies), desc='building passages...'):
        case = case_bodies[i]
        if type(case) == str and len(case) > 0:
            words = case.split(" ")
        else:
            continue
        # case_labels, case_pos = extract_citations(case)

        p_start = 0
        while p_start + max_len < len(words):
            # passages_cites = []
            # for i in range(len(case_labels)):
            #     if case_pos[i][0] >= p_start and case_pos[i][1] <= p_start + max_len:
            #         passages_cites.append(case_labels[i])
            # total_passages_cites.append(passages_cites)
            p = " ".join(words[p_start: p_start+max_len])
            passages.append(p)
            p_start += sw_len
            doc_ids.append(case_ids[i])

        # join the rest
        if p_start < len(words) - 1:
            p = " ".join(words[p_start:])
            passages.append(p)
            doc_ids.append(case_ids[i])
            # passages_cites = []
            # for i in range(len(case_labels)):
            #     if case_pos[i][0] >= p_start and case_pos[i][1] <= len(words):
            #         passages_cites.append(case_labels[i])
            # total_passages_cites.append(passages_cites)
            
    return doc_ids, passages

def build_passages(case_ids, case_bodies, max_len=400):
    passages = []
    doc_ids = []
    for i in trange(len(case_bodies), desc='building passages...'):
        case = case_bodies[i]
        p = []
        p_len = 0
        sents = sent_tokenize(case)
        for sent in sents:
            s_len = len(sent.split(" "))
            # if met the max_len passage threshold
            if s_len + p_len > max_len:
                passages.append(" ".join(p))
                # load the current sent into next paragraph
                p = [sent]
                p_len = s_len
                doc_ids.append(case_ids[i])
            # if not at threshold then keep loading
            else:
                p_len += s_len
                p.append(sent)
        # load the rest
        if len(p) > 0:
            passages.append(" ".join(p))
            doc_ids.append(case_ids[i])
    return doc_ids, passages

def build_passage_collection(data_path, save_path = "/srv/local1/bhou4/case.law.passages-sw.tsv"):
    doc_with_citations = read_tsv(data_path, names = ['case_id', 'case_body', 'citation'])
    case_ids = doc_with_citations['case_id'].tolist()
    case_bodies = doc_with_citations['case_body'].tolist()

    doc_ids, passages = build_passages_sw(case_ids, case_bodies)
    # delete the original data in memory to save space
    del(doc_with_citations)
    del(case_bodies)
    new_passages = []
    for p in passages:
        new_passages.append(p.replace('\n', ' '))
    pids = list(range(len(doc_ids)))

    # save the collection
    with open(save_path, 'w') as jsonl_f:
        for i, text in tqdm(enumerate(new_passages)):
            pid = pids[i]
            if i % 1000000 == 0:
                print(f'Converted {(i // 1000 // 1000)}M', end='\n', flush=True)
            json_obj = {"id": pid, "contents": text}
            jsonl_f.write(json.dumps(json_obj) + "\n")
    # save the pid2did mapping
    d = {'pid': pids, 'doc_id': doc_ids}
    data = pd.DataFrame(data=d)
    save_path = save_path.replace('passages', 'pid2did')
    data.to_csv(save_path, sep='\t', header=False, index=False)


