from sentence_transformers import CrossEncoder
import argparse
from tqdm import tqdm
import pickle
from data_utils import sread_tsv, read_qrels, read_did2pid

def build_qrels_pids(args):
    qids, queries = sread_tsv(args.queries)
    qrels_doc = read_qrels(args.qrels_doc)
    did2pid = read_did2pid(args.did2pid)
    qrels_pids = {}
    for qid, query in zip(qids, queries):
        did = qrels_doc[qid]
        if did in did2pid:
            pids = did2pid[did]
            qrels_pids[qid] = pids
    return qrels_pids

    
def build_rerank_qrels_for_CE(args):
    pids, texts = sread_tsv(args.passage_collection)
    qids, queries = sread_tsv(args.queries)
    qrels_doc = read_qrels(args.qrels_doc)
    did2pid = read_did2pid(args.did2pid)
    
    qrels_passage = {}
    qrels_pids = {}
    for qid, query in zip(qids, queries):
        did = qrels_doc[qid]
        if did in did2pid:
            pids = did2pid[did]
            passages = [texts[int(pid)] for pid in pids]
            qrels_passage[qid] = [(query, passage) for passage in passages]
            qrels_pids[qid] = pids

    rerank_qid_path = args.qrels_doc.replace('qrels-doc_id.tsv', f'qrels-for-CE.pkl')
    with open(rerank_qid_path, 'wb') as f:
        pickle.dump(qrels_passage, f)
    print(f"Saved rerank full qrels to {rerank_qid_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("passage_collection", type=str, help="Path to the passage collection")
    parser.add_argument("queries", type=str, help="Path to the queries")
    parser.add_argument("qrels_doc", type=str, help="Path to the qrels_doc")
    parser.add_argument("did2pid", type=str, help="Path to the did2pid mapping")
    parser.add_argument("--rerank_qrels", type=str, help="Path to the rerank qrels")
    args = parser.parse_args()
    if args.rerank_qrels is None:
        build_rerank_qrels_for_CE(args)
    else:
        with open(args.rerank_qrels, 'rb') as f:
            qrels_passage = pickle.load(f)
        qrels_pids = build_qrels_pids(args)
        
        
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=1024)
    rerank_qrels = {}
    total_len = 0
    for qid, qrels in tqdm(qrels_passage.items(), total=len(qrels_passage), desc='Reranking'):
        scores = model.predict(qrels)
        rerank_qrels[qid] = [qrels[i] for i in range(len(qrels)) if scores[i] > 0.5]
        # sort qrels_pid by score
        temp_pids = [qrels_pids[qid][i] for i in range(len(qrels)) if scores[i] > 0.5]
        qrels_pids[qid] = [pid for _, pid in sorted(zip(scores, temp_pids), reverse=True)]
        total_len += len(rerank_qrels[qid])
    print(f"Average number of passages after reranking: {total_len / len(qids)}")
    save_path = args.qrels_doc.replace('qrels-doc_id', f'qrels-rerank')

    for qid, pids in qrels_pids.items():
        with open(save_path, 'a') as f:
            for pid in pids:
                f.write(f"{qid}\t0\t{pid}\t1\n")



