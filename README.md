This is the repo for CLERC [In-Progress].

### Building Passage-level Triples
1. Build a set of queries used for training, make sure that it is not contaminated by `select_train.py`
2. To rerank the positive passages, build a collection consist of only positive passages of the train queries:
   
   `python pipeline.py build_pos_train_collection_rerank PASSAGE_COLLECTION_PATH --mapping_path DID2PID_PATH --qrels_path --QRELS_DOC_PATH`
4. Rerank with a cross-encoder or BM25 on this positive-only collection, obtain the trec file of search results in TREC format
5. Build a reranked passage-level qrels that only contain the most relevant 1 or 2 positive passages per query:
   
   `python pipeline.py build_rerank_qrels TREC_PATH --qrels_path QRELS_PASSAGE_PATH`
7. Build a set of training triples with a trec file of highly ranked passages and a reranked passage-level qrels:
   
   `python pipeline.py build_triples TREC_PATH --qrels_path QRELS_PASSAGE_PATH`

## Cite CLERC
@article{abe2024clerc,
  title={CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation},
  author={Abe Bohan Hou, Orion Weller, Guanghui Qin, Eugene Yang, Dawn Lawrie, Nils Holzenberger, Andrew Blair-Stanek, Benjamin Van Durme},
  journal={ArXiv},
  year={2024},
  url={https://arxiv.org/pdf/2406.17186}
}
