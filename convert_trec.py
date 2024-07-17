import argparse
from data_utils import read_mapping, read_trec
from tqdm import tqdm
from queue import PriorityQueue

def aggregate_doc_scores_max_k(trec, all_qids, top_k = 1000):
    for qid in tqdm(trec, total=len(trec), desc="Aggregating..."):
        for did, score in trec[qid]:
            if qid not in all_qids:
                all_qids[qid] =  PriorityQueue(maxsize=top_k)
            if all_qids[qid].qsize() >= top_k:
                all_qids[qid].get()
            all_qids[qid].put((float(score), did))
    return all_qids

def tsv_to_trec(path):
    with open(path, "r") as f:
        lines = f.readlines()
        f.close()
    trec = {}
    for line in lines:
        qid, did, score = line.strip().split("\t")
        if qid not in trec:
            trec[qid] = [(did, score)]
        else:
            trec[qid].append((did, score))
    return trec

def maxp_trec(trec, pid2did, top_k = 1000):
    all_qids = {}
    all_qids_did_set = {}
    for qid in tqdm(trec, total=len(trec), desc="Aggregating..."):
        for pid, score in trec[qid]:
            did = pid2did[pid]
            if qid not in all_qids:
                all_qids[qid] =  PriorityQueue(maxsize=top_k)
                all_qids_did_set[qid] = set()
            # pop out if it's greater than top_k
            if all_qids[qid].qsize() >= top_k:
                all_qids[qid].get()
                all_qids[qid].put((float(score), did))
            # normal add if it's not in the set and less than top_k
            elif did not in all_qids_did_set[qid]:
                all_qids_did_set[qid].add(did)
                all_qids[qid].put((float(score), did))
            # do maxP if it's already in the set
            elif did in all_qids_did_set[qid]:
                prev = all_qids[qid].get(did) 
                if prev[0] < float(score):
                    all_qids[qid].put((float(score), did))
                else:
                    all_qids[qid].put(prev)
    return all_qids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_path')
    parser.add_argument("--tsv_path", type=str)
    parser.add_argument("--trec_path", type=str, nargs='+')
    args = parser.parse_args()
    trec_paths = args.trec_path
    if args.trec_path and ".trec" in trec_paths[0]:
        save_path = trec_paths[0].replace(".trec", "_aggregated.trec") 
    elif args.tsv_path and ".tsv" in args.tsv_path:
        save_path = args.tsv_path.replace(".tsv", "_aggregated_mapped.tsv")
    elif args.tsv_path and ".txt" in args.tsv_path:
        save_path = args.tsv_path.replace(".txt", "_aggregated_mapped.txt")
        
    all_qids = {}

    if args.tsv_path and not args.trec_path:
        trec = tsv_to_trec(args.tsv_path)
        print("Converted tsv to trec")
        all_qids = maxp_trec(trec, read_mapping(args.mapping_path))
    elif args.tsv_path:
        raise ValueError("Cannot provide both tsv_path and trec_path")
    else:
        for path in trec_paths:
            trec = read_trec(path, args.mapping_path)
            all_qids = aggregate_doc_scores_max_k(trec, all_qids)
            print(f"Finished aggregating {path}")

    with open(save_path, "w") as f:
        for qid, pq in tqdm(all_qids.items(), "Writing"):
            scores, dids = [], []
            while not pq.empty():
                score, did = pq.get()
                scores.append(score)
                dids.append(did)
            scores.reverse()
            dids.reverse()
            for idx, (score, did) in enumerate(zip(scores, dids)):
                f.write(f"{qid} Q0 {did} {idx+1} {score} PLAID\n")
        f.close()
    
    
    
            


        
    