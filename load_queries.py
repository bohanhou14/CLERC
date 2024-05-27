from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
import pickle
from time import time
import re
from data_utils import extract_citations, build_collection, match_by_token_overlap, clean_cstr
import random
import numpy as np
fdq = chr(8220)
bdq = chr(8221)
random.seed(42)

def find_long_p(text, p):
    res = []
    for m in re.finditer(p, text):
        words = text[m.start():m.end()].split()
        if len(words) >= 5:
            res.append((m.start(), m.end()))
    return res

def extract_dq_from_doc(text, cites, window = 300, small_window=150):
    p = f'(?<={fdq})(.*?)(?={bdq})'
    quotes_idx = find_long_p(text, p)
    # p = "Id." # exact match Id. 
    # ids_idx = np.array([d[0] for d in find_all_p(text, p)])
    if len(quotes_idx) == 0:
        return None
    
    cites_indices = [text.index(c) for c in cites]
    indices = np.argsort(cites_indices)
    # sort both
    cites_indices = [cites_indices[i] for i in indices]
    cites = [cites[i] for i in indices]

    i, j = 0, 0
    found_pairs = {}

    def find_dis(cite, quote_start, quote_end):
        return min(abs(cite - quote_start), abs(cite - quote_end))
    
    def is_closest_cite(cite, quote_start, quote_end, next_cite):
        dis = find_dis(cite, quote_start, quote_end)
        next_dis = find_dis(next_cite, quote_start, quote_end)
        return dis < next_dis 
    
    while j < len(cites_indices):
        if i >= len(quotes_idx):
            return found_pairs
        quote_start = quotes_idx[i][0]
        quote_end = quotes_idx[i][1]
        cite_idx = cites_indices[j]
        cite = cites[j]
        dis = find_dis(cite_idx, quote_start, quote_end)
        closest_cite = is_closest_cite(cite_idx, quote_start, 
                                       quote_end, cites_indices[min(j+1, len(cites_indices)-1)])
        if closest_cite and dis > window:
            i += 1 # probably do not have a citation!
        elif closest_cite:
            if cite in found_pairs:
                found_pairs[cite].append((text[quote_start:quote_end], dis))
            else:
                found_pairs[cite] = [(text[quote_start:quote_end], dis)]
            i += 1 # found the direct quote citation, move on to next one
            if text[quote_end-1] == ',':
                if i < len(quotes_idx)-1:
                    next_quote_start = quotes_idx[i+1][0]
                    next_quote_end = quotes_idx[i+1][1]
                    dis = find_dis(cite_idx, next_quote_start, next_quote_end)
                    found_pairs[cite].append((text[next_quote_start:next_quote_end], dis))
                    j += 1; i += 1
            else:
                j += 1 # move on to the next case
        else:
            j += 1 # move on to the next case

    # filter the probably wrong ones
    found_pairs = {k: [x for x in v if x[1] <= small_window] 
                   for k, v in found_pairs.items()}
    found_pairs = {k: v for k, v in found_pairs.items()
                   if len(v) > 0 }
    return found_pairs

def build_queries(cites_path = "cites2did.pkl", case_path = "case.law.citations.tsv", save_path = None, keep_citation=False, n=None, window_size=150):
    with open(cites_path, 'rb') as f:
        d = pickle.load(f)
    d = dict(d)
    # all citation names
    citation_names = list(d.keys())
    name_list = [clean_cstr(n) for n in citation_names]
    t0 = time()
    df = pd.read_csv(case_path, sep='\t', names=['doc_id', 'text'])
    texts = df['text'].tolist()
    texts = [str(t).replace("\n", " ") for t in texts]
    print(f"Read tsv after {time() - t0} seconds")
    random.shuffle(texts)
    texts = texts[:n]
    total_queries = [] ; total_ans_ids = [] ; total_citations = []; total_direct_quotes = []
    i = 0
    init_load = False
    qids = []
    for text in tqdm(texts):
        queries, ans_ids, citations, direct_quotes = build_queries_one_case(text, name_list, citation_names, d, keep_citation=keep_citation, window_size=window_size)
        if len(queries) == 0:
            continue
        total_queries.extend(queries)
        total_ans_ids.extend(ans_ids)
        total_citations.extend(citations)
        total_direct_quotes.extend(direct_quotes)
        i += 1
        # save
        if i % 1000 == 0:
            if not init_load:
                qids = list(range(len(total_queries)))
                data = {
                    'qid': qids,
                    'text': total_queries,
                    'ans': total_ans_ids,
                    'citations': total_citations,
                    'dq': total_direct_quotes
                }
                pd.DataFrame(data=data).to_csv(save_path, sep = '\t', header=False, index=False)
                total_queries = []
                total_ans_ids = []
                total_citations = []
                total_direct_quotes = []
                del(data)
                init_load = True
            else:
                start_qid = qids[-1] + 1
                qids = list(range(start_qid, start_qid + len(total_queries)))
                data = {
                    'qid': qids,
                    'text': total_queries,
                    'ans': total_ans_ids,
                    'citations': total_citations,
                    'dq': total_direct_quotes
                }
                # append data
                pd.DataFrame(data=data).to_csv(save_path, mode='a', sep = '\t', header=False, index=False)
                total_queries = []
                total_ans_ids = []
                total_direct_quotes = []
                total_citations = []
                del(data)
    if len(total_queries) > 0 and len(qids) > 0:
        start_qid = qids[-1] + 1
        qids = list(range(start_qid, start_qid + len(total_queries)))
        data = {
            'qid': qids,
            'text': total_queries,
            'ans': total_ans_ids,
            'citations': total_citations,
            'dq': total_direct_quotes
        }
        pd.DataFrame(data=data).to_csv(save_path, mode='a', sep = '\t', header=False, index=False)
        del(data)

def build_query(label, text, keep_citation=True, window_size=150):
    new_label = "".join(label.split(" "))
    new_text = text.replace(label, " " + new_label + " ")
    words = text.split(" ")
    new_words = new_text.split(" ")
    try:
        label_pos = new_words.index(new_label)
    except:
        print(label)
        return -1
    start = max(0, label_pos-window_size)
    if keep_citation == True:
        end = min(len(words), label_pos+window_size)
        sentence = words[start:end]
        # print(sentence)
        # print(1)
    else:
        end = min(len(new_words), label_pos+window_size)
        # remove the pincite
        if re.match(r"\d+,", new_words[label_pos+2]):
            sentence = new_words[start:label_pos] + new_words[(label_pos + 1):(label_pos+2)] + new_words[(label_pos + 3):end]
        else:
            sentence = new_words[start:label_pos] + new_words[(label_pos + 1):]
    query = " ".join(sentence)
    return query

def clean_query(query, case_labels):
    for c in case_labels:
        query = query.replace(c, " ")
    query = " ".join(query.split())
    return query

def build_queries_one_case(text, name_list, citation_names, d, keep_citation=False, window_size=150):
    try:
        case_labels, case_pos = extract_citations(text)
    except:
        return -1, -1
    
    found_pairs = extract_dq_from_doc(text, case_labels, window=300, small_window=150)
        
    clean_case_labels = [clean_cstr(l) for l in case_labels]
    # ids of unique citations
    unique_ids = set()
    queries = []
    ans_ids = []
    citations = []
    direct_quotes = []
    # resolve the duplicate case_labels extracted from text
    for i in range(len(clean_case_labels)):
        # find the citation name that is approximately the same
        # breakpoint()
        l = clean_case_labels[i]
        # match_id = match_by_token_overlap(l, name_list)
        # skip if found nothing
        if not (l in name_list):
            continue
        else:
            match_id = name_list.index(l)
        match = citation_names[match_id]
        # only add when the match_ID is unique
        if d[match] not in unique_ids:
            # unique case label
            uc = case_labels[i]
            # unique case label id
            ucid = d[match]
            # add ucid to the set
            unique_ids.add(ucid)
            query = build_query(uc, text, keep_citation=keep_citation, window_size=window_size)
            if query == -1:
                continue
            queries.append(query) 
            ans_ids.append(ucid)  
            citations.append(uc)
            if found_pairs != None and uc in found_pairs:
                direct_quotes.append(found_pairs[uc])
            else:
                direct_quotes.append(None)


    if len(queries) != len(citations):
        print(len(queries), len(citations))
        print(f"queries: {queries}")
        print(f"citations: {citations}")
        assert 1 == 0
    return queries, ans_ids, citations, direct_quotes
        
# qids = list(range(len(total_queries)))

# data = {
#     'qid': qids,
#     'text': total_queries,
#     'ans': total_ans_ids 
# }
# pd.DataFrame(data=data).to_csv("data/case.law.queries.tsv", sep = '\t', header=False, index=False)



# print(case_labels)
# extracted_cites = extract_citations(text)





    






