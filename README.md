# Official Repo for CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation

## Introduction
This repo supports the development of CLERC, a dataset for legal case retrieval and retrieval-augmented analysis generation built on Caselaw Access Project (CAP) (kudos to https://case.law !). Feel free to read our paper at: https://arxiv.org/pdf/2406.17186

Our contributions in the paper are threefold:
1. Through working with legal professionals, we provide a formulation of legal case retrieval and generation tasks that balance the needs and perspectives of legal professionals with computational feasibility.

2. We build an open-source pipeline for transforming CAP into a large-scale, high-quality dataset designed for training and evaluating models on legal IR and RAG tasks.

3. We conduct a comprehensive evaluation of long-context case retrieval and retrieval-augmented legal analysis generation on CLERC with state-of-the-art methods, revealing that IR models struggle to retrieve relevant documents and LLMs frequently hallucinate.

In this repo, we provide codes for replicating the curation process of CLERC datasets as well as running retrieval and generation experiments. We will explain the two main subsets of CLERC, **retrieval** and **generation** separately.

## Retrieval

## Generation
Please refer to our paper for the formulation of task and README in the generation subdirectory (https://github.com/bohanhou14/CLERC/blob/main/generation/README.md)
## Cite CLERC
```
@article{abe2024clerc,
  title={CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation},
  author={Abe Bohan Hou and Orion Weller and Guanghui Qin and Eugene Yang and Dawn Lawrie and Nils Holzenberger and Andrew Blair-Stanek and Benjamin Van Durme},
  journal={ArXiv},
  year={2024},
  url={https://arxiv.org/pdf/2406.17186}
}
```


## Building Passage-level Triples
1. Build a set of queries used for training, make sure that it is not contaminated by `select_train.py`
2. To rerank the positive passages, build a collection consist of only positive passages of the train queries:
   
   `python pipeline.py build_pos_train_collection_rerank PASSAGE_COLLECTION_PATH --mapping_path DID2PID_PATH --qrels_path --QRELS_DOC_PATH`
4. Rerank with a cross-encoder or BM25 on this positive-only collection, obtain the trec file of search results in TREC format
5. Build a reranked passage-level qrels that only contain the most relevant 1 or 2 positive passages per query:
   
   `python pipeline.py build_rerank_qrels TREC_PATH --qrels_path QRELS_PASSAGE_PATH`
7. Build a set of training triples with a trec file of highly ranked passages and a reranked passage-level qrels:
   
   `python pipeline.py build_triples TREC_PATH --qrels_path QRELS_PASSAGE_PATH`

