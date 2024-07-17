import pandas as pd
from CLERC.data.src.data_utils import extract_citations, build_collection, match_by_token_overlap, clean_cstr
import os
from tqdm import trange
import numpy as np
dd = os.environ['DATADIR']
# metadata = pd.read_csv(f"{dd}/case.law.data/metadata.csv", sep=',')
# doc_ids = metadata['id'].tolist()
np.random.seed(42)

def sample_queries_and_build_qrels(queries_path, pid2did_path, k=-1):
    if "indirect" in queries_path:    
        queries_df = pd.read_csv(queries_path, sep='\t', names=['qid', 'text', 'ans_id', 'citations', 'removed_sents'])
    elif "direct" in queries_path:
        queries_df = pd.read_csv(queries_path, sep='\t', names=['qid', 'text', 'ans_id', 'citations', 'removed_sents', 'quote'])  

    did2pid_path = pid2did_path.replace('pid2did', 'did2pid')
    pid2did = pd.read_csv(pid2did_path, sep='\t', names=['pid', 'doc_id'])
    pids = pid2did['pid'].tolist()
    dids = pid2did['doc_id'].tolist()
    dids_set = set(dids)
    if not os.path.exists(did2pid_path):
        did2pid = {}
        for i in range(len(pids)):
            if dids[i] not in did2pid:
                did2pid[dids[i]] = [pids[i]]
            else:
                did2pid[dids[i]].append(pids[i])
        did2pid_path = pid2did_path.replace('pid2did', 'did2pid')
        data = pd.DataFrame(data=did2pid.items(), columns=['doc_id', 'pid'])
        data.to_csv(did2pid_path, sep='\t', header=False, index=False)
        print(f"Saved did2pid mapping to {did2pid_path}")
    else:
        # load did2pid if it exists
        did2pid_df = pd.read_csv(did2pid_path, sep='\t', names=['doc_id', 'pid'])
        did2pid = {}
        doc_ids = did2pid_df['doc_id'].tolist()
        pids = did2pid_df['pid'].tolist()
        for i in range(len(doc_ids)):
            did2pid[doc_ids[i]] = pids[i].strip('][').split(', ')
            
    qrels_path = queries_path.replace('.tsv', f'-qrels.tsv')
    doc_qrels_path = qrels_path.replace('.tsv', f'-doc_id.tsv')
    qrels = []
    qids = queries_df['qid'].tolist()
    queries = queries_df['text'].tolist()
    ans_ids = queries_df['ans_id'].tolist()
    # breakpoint()
    # filter queries whose answers do not appear in the corpus
    qids = [qids[i] for i in range(len(ans_ids)) if ans_ids[i] in dids_set]
    queries = [queries[i] for i in range(len(ans_ids)) if ans_ids[i] in dids_set]
    ans_ids = [ans_ids[i] for i in range(len(ans_ids)) if ans_ids[i] in dids_set]
    if k == -1:
        k = len(ans_ids)
    # sample k queries
    indices = np.random.choice(len(ans_ids), k, replace=False)
    qids = [qids[i] for i in indices]
    queries = [queries[i] for i in indices]
    ans_ids = [ans_ids[i] for i in indices]
    new_queries_df = pd.DataFrame(data={'qid': qids, 'text': queries})
    new_queries_path = queries_path.replace('.tsv', f'-topics-k={k}.tsv')
    new_queries_df.to_csv(new_queries_path, sep='\t', header=False, index=False)
    if "indirect" not in queries_path:
        quotes = queries_df['quote'].tolist()
        quotes = [quotes[i] for i in indices]
        quote_path = queries_path.replace('.tsv', f'-quote-k={k}.tsv')
        pd.DataFrame(data={'qid': qids, 'text': quotes}).to_csv(quote_path, sep='\t', header=False, index=False)   
    doc_qrels = []
    
    for i in trange(len(ans_ids)):
        aid = ans_ids[i]
        doc_qrels.append(f"{qids[i]}\t0\t{aid}\t1")
        try:
            temp_pids = did2pid[aid]
        except:
            continue
        for pid in temp_pids:
            qrels.append(f"{qids[i]}\t0\t{pid}\t1")

    with open(qrels_path, 'w') as f:
        for qrel in qrels:
            f.write(f"{qrel}\n")
        f.close()
    print(f"Saved qrels to {qrels_path}")

    with open(doc_qrels_path, 'w') as f:
        for qrel in doc_qrels:
            f.write(f"{qrel}\n")
        f.close()
    print(f"Saved doc qrels to {doc_qrels_path}")

    
        

    


