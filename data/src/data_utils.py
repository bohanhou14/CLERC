import lzma
import json
import sys
from tqdm import tqdm, trange
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from eyecite import get_citations, clean_text, resolve_citations, annotate_citations
from eyecite.models import FullCaseCitation, Resource, CaseCitation
from eyecite.resolve import resolve_full_citation
from eyecite.tokenizers import HyperscanTokenizer
import re
ENDPUNCTS = ['.','!', ')', '"', '?']

def read_trec(path):
    with open(path, "r") as f:
        lines = f.readlines()
        f.close()
    trec = {}
    for line in lines:
        qid, _, pid, _, score, _ = line.strip().split(" ")
        if qid not in trec:
            trec[qid] = [(pid, score)]
        else:
            trec[qid].append((pid, score))
    return trec


def take_max_k(path, k):
    with open(path, "r") as f:
        lines = f.readlines()
        f.close()
    path = path.replace(".tsv", f"_{k}.tsv")
    with open(path, "w") as f:
        for line in lines[:k]:
            f.write(line)
        f.close()

def read_qrels_doc(path):
    qrels = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Reading qrels', total=len(lines)):
            qid, _, did, _ = line.strip().split()
            qrels[qid] = did
    return qrels

def read_mapping(path):
    mapping = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Reading mapping', total=len(lines)):
            pid, did = line.strip().split()
            mapping[pid] = did
    return mapping

def read_qrels_passage(path):
    qrels = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Reading qrels', total=len(lines)):
            qid, _, pid, _ = line.strip().split()
            if qid not in qrels:
                qrels[qid] = set([pid])
            else:
                qrels[qid].add(pid)
    return qrels

def read_did2pid(path):
    did2pid = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Reading did2pid', total=len(lines)):
            did, pids = line.split("\t")
            pids = pids.strip().strip('][').split(', ')
            did2pid[did] = pids
    return did2pid

# speed read tsv file line by line
def sread_tsv(path, k=None):
    collections = []
    pids = []
    with open(path) as f:
        i = 0
        for line in f:
            pid, text, *rest = line.strip().split("\t")
            collections.append(text)
            pids.append(pid)
            i += 1
            if i % 1000000 == 0:
                print(f'Loaded {(i // 1000 // 1000) }M', end='\n', flush=True)
            if k != None and i == k:
                break
        f.close()
    return pids, collections

def clean_cstr(c):
    # remove spaces and dots
    l = re.sub(r'\W+', '', c)
    return l

def read_tsv(path, names=None):
    if names == None:
        if 'passages' in path:
            names = ['pid', 'text']
        elif 'queries' in path:
            names = ['qid', 'text', 'ans_id']
        else:
            raise NotImplementedError("neither passages nor queries")
    df = pd.read_csv(path, sep='\t', names=names)
    return df

def build_collection():
    with lzma.open("data/data.jsonl.xz") as f:
        case_ids, case_bodies, citations = [], [], []
        for line in tqdm(f):
            case = json.loads(str(line, 'utf8'))
            case_ids.append(case['id'])
            body = []
            for item in case['casebody']['data']['opinions']:
                body.append(item['text'])
            cites = []
            case_bodies.append(" ".join(body))
            with open('temp.txt', 'w') as sys.stdout:
                print(f"case: {case}")
            for c in case['citations']:
                cites.append(c['cite'])
                if c['type'] == 'official' and len(cites) > 1:
                    # swap the official to the first entry
                    cites[0], cites[-1] = cites[-1], cites[0]
            citations.append(cites)
            break
    return case_ids, case_bodies, citations

def clean_paragraph(text):
    sents = sent_tokenize(text)
    new_text = []
    if sents[0].isupper():
        new_text.append(sents[0])
    if len(sents) > 1:
        new_text.append(" ".join(sents[1:-1]))
    if sents[-1][-1] in ENDPUNCTS:
        new_text.append(sents[-1])
    return " ".join(new_text)

def match_by_token_overlap(case_name, name_list):
    for i in range(len(name_list)):
        name = name_list[i]
        if name != case_name:
            continue
        else:
            return i
    return -1

def extract_citations(text):
    # print(text)
    text = clean_text(text, ['html', 'all_whitespace'])
    # tokenizer = HyperscanTokenizer(cache_dir='.test_cache')
    citations = get_citations(text)
    citations = list(citations)
    resolutions = resolve_citations(citations)
    cases = [res for res in resolutions if (isinstance(res.citation, FullCaseCitation) or isinstance(res.citation, CaseCitation))]
    case_labels = [c.citation.token.data for c in cases]
    case_pos = [(c.citation.token.start, c.citation.token.end) for c in cases]
    return case_labels, case_pos

if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'take_max_k':
        path = sys.argv[2]
        k = int(sys.argv[3])
        take_max_k(path, k)