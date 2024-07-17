import sys
from CLERC.data.src.data_utils import sread_tsv

if __name__ == '__main__':
    # read in one argument for all the queries
    # and read in another argument for the test queries
    # select the train and validation queries
    # and write them to a file
    train_name = sys.argv[1]
    test_name = sys.argv[2]
    train_qids, train_texts = sread_tsv(train_name)
    test_qids, test_texts = sread_tsv(test_name)
    new_train_qids, new_train_texts = [], []
    for qid, text in zip(train_qids, train_texts):
        if qid not in test_qids:
            new_train_qids.append(qid)
            new_train_texts.append(text.replace("\n", " "))
    new_train_name = train_name.replace('.tsv', '_train.tsv')
    
    with open(new_train_name, 'w') as f:
        for qid, text in zip(new_train_qids, new_train_texts):
            f.write(f'{qid}\t{text}\n')
        f.close()
    

