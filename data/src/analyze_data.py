import argparse
import pandas as pd
from tqdm import trange
from CLERC.data.src.data_utils import extract_citations, sread_tsv
from matplotlib import pyplot as plt

def analyze_citation_density(args):
    data = pd.read_csv(args.data, sep='\t', names=['did', 'text'])
    decile_buckets = {i: [] for i in range(1, 11)}
    for i in trange(len(data), desc='Processing data'):
        case = data.iloc[i]
        text = case['text']
        total_len = len(text)
        paragraphs = text.split('\n')
        p_middle = [len(paragraph) // 2 for paragraph in paragraphs]
        p_start = []
        for a in range(len(paragraphs)):
            if a == 0:
                p_start.append(0)
            else:
                p_start.append(p_start[a-1] + len(paragraphs[a-1]))

        p_pos = [p_start[b] + p_middle[b] for b in range(len(paragraphs))]
       
        deciles_cutoff = [int(total_len * c / 10) for c in range(0, 10)]

        case_labels, case_pos = extract_citations(text)
        p_cite_count = []
        for j in range(len(paragraphs)):
            paragraph = paragraphs[j]
            count = 0
            for k in range(len(case_labels)):
                label = case_labels[k]
                if label in paragraph:
                    count += 1
            p_cite_count.append(count)

        p_word_len = [len(paragraph.split(" ")) for paragraph in paragraphs]
        # find which decile does the paragraph belong to based on p_pos
        for j in range(len(paragraphs)):
            pos = p_pos[j]
            for k in range(0, 9):
                if pos > deciles_cutoff[k] and pos <= deciles_cutoff[k+1]:
                    decile_buckets[k+1].append(p_cite_count[j]/p_word_len[j])
                    break
            if pos > deciles_cutoff[9]:
                decile_buckets[10].append(p_cite_count[j]/p_word_len[j])

    for i in range(1, 11):
        print(f"Decile {i}: {sum(decile_buckets[i])/len(decile_buckets[i])}")
    # plot
    plt.plot([i for i in range(1, 11)], [sum(decile_buckets[i])/len(decile_buckets[i]) for i in range(1, 11)])
    plt.xlabel('Decile')
    plt.ylabel('Citation Density')
    plt.title('Citation Density in Deciles')
    plt.savefig('decile_cite_density.png')

def analyze_size_dist(args):
    dids, texts = sread_tsv(args.data)
    sizes = []
    for text in texts:
        sizes.append(len(text.split(" ")))
    print("max, min, avg")
    print(max(sizes), min(sizes), sum(sizes)/len(sizes))
    plt.hist(sizes, bins=100)
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.title('Size Distribution')
    plt.savefig('size_dist.png')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['citation_density', "size_dist"], help="task to perform")
    parser.add_argument("data")
    args = parser.parse_args()
    if args.task == 'citation_density':
        analyze_citation_density(args)
    elif args.task == 'size_dist':
        analyze_size_dist(args)
    else:
        raise ValueError(f"Invalid task {args.task}")





    