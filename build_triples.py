import argparse
from tqdm import tqdm, trange
from data_utils import sread_tsv, read_qrels_doc, read_did2pid, read_trec, read_qrels_passage

def build_pos_train_collection_rerank(passage_collection_path, did2pid_path, qrels_doc_path):
    pids, texts = sread_tsv(passage_collection_path)
    qrels_doc = read_qrels_doc(qrels_doc_path)
    did2pid = read_did2pid(did2pid_path)

    pos_passage_collection = []
    pos_pids_set = set()
    
    for qid, did in tqdm(qrels_doc.items(), total=len(qrels_doc), desc='Building positive passage collection...'):
        if did in did2pid:
            pids = did2pid[did]
            passages = []
            for pid in pids:
                if pid not in pos_pids_set:
                    pos_pids_set.add(pid)
                    passages.append(f"{pid}\t{texts[int(pid)]}")
            pos_passage_collection.extend(passages)

    qrel_doc_head = qrels_doc_path.split('/')[-1].replace('.tsv', '')
    
    with open(passage_collection_path.replace('.tsv', f'-{qrel_doc_head}.tsv'), 'w') as f:
        for p in pos_passage_collection:
            f.write(f"{p}\n")
        f.close()

def build_rerank_triples(trec_path, qrels_passage_path):
    trec = read_trec(trec_path)
    qrels = read_qrels_passage(qrels_passage_path)

    reranked_qrels = {}
    for qid in tqdm(trec, desc='Building qrels...', total=len(trec)):
        trec_pids = [pid for pid, _ in trec[qid]]
        for pid in trec_pids:
            if qid in qrels and pid in qrels[qid]:
                if qid not in reranked_qrels:
                    reranked_qrels[qid] = [pid]
                else:
                    reranked_qrels[qid].append(pid)
    # debug
    
    with open(args.qrels_passage_path.replace('.tsv', '-reranked.tsv'), 'w') as f:
        for qid in reranked_qrels:
            # top-2 passage because of sliding window, more than 1 passage can be relevant
            for i in range(min(len(reranked_qrels[qid]), 2)):
                f.write(f"{qid}\t0\t{reranked_qrels[qid][i]}\t1\n")
        f.close()
    
    print(f"percentage of queries found at least a relevant passage: {len(reranked_qrels.keys()) / len(trec.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("passage_collection", type=str, help="Path to the passage collection")   
    parser.add_argument("did2pid", type=str, help="Path to the did2pid mapping")
    parser.add_argument("qrels_doc", type=str, help="Path to the qrels")
    args = parser.parse_args()

    pids, texts = sread_tsv(args.passage_collection)
    qrels_doc = read_qrels_doc(args.qrels_doc)
    did2pid = read_did2pid(args.did2pid)

    # collect only the positive passages
    build_pos_train_collection_rerank(args.passage_collection, args.did2pid, args.qrels_doc)

        


    


    
