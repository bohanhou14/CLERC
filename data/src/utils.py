import argparse
from CLERC.data.src.data_utils import sread_tsv

def shorten_tsv(data, k):
    qids, texts = sread_tsv(data)
    new_qids = qids[:int(k)]
    new_texts = texts[:int(k)]

    save_path = data + f"-k={k}.tsv"
    with open(save_path, 'w') as f:
        for qid, text in zip(new_qids, new_texts):
            f.write(f"{qid}\t{text}\n")
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the data")
    parser.add_argument("k", type=str, help="Number of rows to keep")
    args = parser.parse_args()
    
    shorten_tsv(args.data, args.k)