import argparse
from build_collections import build_citations_tsv, build_passage_collection
from load_queries import build_queries
from filter_queries import filter_queries
from build_qrels import sample_queries_and_build_qrels
from build_triples import build_pos_train_collection_rerank, build_rerank_triples
import os
dd = os.environ['DATADIR']

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices = ['process_raw', 'build_collections', 
    'build_queries', 'build_qrels', 'filter_queries', 'filter_queries_all', 'build_pos_train_collection_rerank', 'build_rerank_triples'])
    parser.add_argument("data", type=str, help="path to the dataset")
    parser.add_argument("--trec_path", type=str, help="path to trec file")
    parser.add_argument("--qrels_path", type=str, help="path to qrels file")
    parser.add_argument("--queries_path", type=str, help="path to queries.tsv")
    parser.add_argument("--mapping_path", type=str, default="/brtx/606-nvme2/bhou4/CLERC/collection/mapping.pid2did.tsv")
    parser.add_argument("--save_path", type=str, help="path to save data")
    parser.add_argument("--save_dir", type=str, help="dir to save data", default="/brtx/606-nvme2/bhou4/CLERC/collection")
    parser.add_argument("--majority_only", action='store_true', help="whether to only keep the majority class", default=True)
    parser.add_argument("--cites_path", type=str, help="path to cites2did.pkl", default=f"{dd}/case.law.data/cites2did.pkl")
    parser.add_argument("--keep_citation", "-k", help="whether to remove citation in queries", action='store_true')
    parser.add_argument("--n", type=int, help="number of data requested", default=-1)
    parser.add_argument("--window_size", type=int, help="window size for passage collection", default=150)
    args = parser.parse_args()
    return args

def main(args):
    if args.task == 'process_raw':
        print("Processing raw data...")
        # return a df with case_id, text, citations
        # also saves cites2did
        build_citations_tsv(args.data, args.save_dir, args.majority_only)
    elif args.task == 'build_collections':
        print("Building collections...")
        # build collections by breaking down case documents into passages through sliding-window
        build_passage_collection(args.data, save_path=args.save_path)
    elif args.task == 'build_queries':
        # print(args.keep_citation)
        print("Building queries...")
        build_queries(cites_path = args.cites_path, case_path = args.data, save_path = args.save_path, keep_citation=True, n=args.n, window_size=args.window_size)
    elif args.task == 'filter_queries':
        print("Filtering queries...")
        filter_queries(args.data, num_queries=args.n, option='single')
    elif args.task == 'filter_queries_all':
        print("Filtering queries (all citations) ...")
        filter_queries(args.data, num_queries=args.n, option='all')
    elif args.task == 'build_qrels':
        print("Building qrels...")
        sample_queries_and_build_qrels(queries_path=args.data, pid2did_path=args.mapping_path, k=args.n)
        # build_qrels(queries_path=args., passages_path=args.data, pid2did_path=args.pid2did_path, k=args.n)
    elif args.task == 'build_pos_train_collection_rerank':
        print("Building positive training collection for rerank...")
        build_pos_train_collection_rerank(passage_collection_path=args.data, did2pid_path=args.mapping_path, qrels_doc_path=args.qrels_path)
    elif args.task == 'build_rerank_triples':
        print("Building rerank triples...")
        build_rerank_triples(trec_path=args.data, qrels_passage_path=args.qrels_path)

if __name__ == "__main__":
    args = cli()
    main(args)
