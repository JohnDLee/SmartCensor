# Retrieves Baseline statistics

from smartcensor.data import JigsawData, JigsawPath

from detoxify.detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util

import re
import tqdm
import os

def load_toxic_bank():
    with open(os.path.join(os.environ['MAINDIR'], 'data/toxic_bank/toxic_words.txt')) as td:
        toxic_words = [re.sub("\*", "\\\*", w.rstrip()) for w in td]
    return toxic_words

if __name__ == '__main__':
    
    import sys
    mode = sys.argv[1]
    if mode not in {'all', 'toxic', 'nontoxic'}:
        raise ValueError("Argument must be one of all, toxic, or nontoxic")
    
    ## Load Testing Data
    baseline_data = JigsawData(JigsawPath.test)
    zipped_data = baseline_data.get_zipped_data()
    ## Load Toxic Bank
    toxic_words = load_toxic_bank()
    ## Load Toxic Bert
    toxic_metric = Detoxify('original')
    ## Load Sentence embedder
    sentence_emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    ## Trackers
    keys = toxic_metric.predict('')
    reductions = {k: [] for k in keys.keys()}
    original = {k:[] for k in keys.keys()}
    sim = []
    
    for id, comment, labels in tqdm.tqdm(zipped_data, desc='Cleaning Strings'):
        # for every toxic word, sub it for empty
        if mode == 'toxic' and sum(labels) == 0:
            # skip nontoxic
            continue
        elif mode == 'nontoxic' and sum(labels) > 0:
            # skip toxic
            continue
        
        clean_string = comment
        for toxic_w in toxic_words:
            clean_string = re.sub(pattern=f"{toxic_w}",
                   repl='',
                   string=clean_string,
                   flgas =re.IGNORECASE)
        # print(comment, clean_string)
        
        ## Oberve Toxicity
        results = toxic_metric.predict([comment, clean_string])
        for k in reductions.keys():
            reductions[k].append(results[k][1])
            original[k].append(results[k][0])

        ## Observe Sentence Cosine Sim
        embedding_toxic = sentence_emb.encode(comment, convert_to_tensor=True)
        embedding_clean = sentence_emb.encode(clean_string, convert_to_tensor=True)
        sim.append(util.pytorch_cos_sim(embedding_toxic, embedding_clean).item())

        
    ## Average Toxicity Reduction
    print("Average Original Values for Toxicity")
    for k, val in original.items():
        print(f'{k:20} | {sum(val)/len(val):.4f}')
    print("Average Detoxified in Toxicity:")
    for k, val in reductions.items():
        print(f'{k:20} | {sum(val)/len(val):.4f}')
    print("Average Reduction in Toxicity:")
    for k in original.keys():
        s = 0
        count = 0
        for v1, v2 in list(zip(original[k], reductions[k])):
            s += v2 - v1
            count += 1
        print(f'{k:20} | {s/count:.4f}')
    
    print(f"Average Cosine Similarity: {sum(sim)/len(sim):.4f}")