"""Simple implementation of single-reference, case-sensitive BLEU
without tokenization."""

import collections
import math

def ngrams(seg, n):
    c = collections.Counter()
    for i in range(len(seg)-n+1):
        c[tuple(seg[i:i+n])] += 1
    return c

def card(c):
    """Cardinality of a multiset."""
    return sum(c.values())

def zero():
    return collections.Counter()

def count(t, r, n=4):
    """Collect statistics for a single test and reference segment."""

    stats = collections.Counter()
    for i in range(1, n+1):
        tngrams = ngrams(t, i)
        stats['guess',i] += card(tngrams)
        stats['match',i] += card(tngrams & ngrams(r, i))
    stats['reflen'] += len(r)
    return stats

def score(test_sents, ref_sents, n=4):
    """Compute BLEU score.

    Arguments:
    - test_sents: test sentences (list of list of strs)
    - ref_sents: reference sentences (list of list of strs)"""

    c = zero()
    for t, g in zip(test_sents, ref_sents):
        c += count(t, g, n=n)

    if c['guess',1] == 0:
        return 0.
        
    b = 1.
    for i in range(1, n+1):
        b *= c['match',i]/c['guess',i] if c['guess',i] > 0 else 0
    b **= 0.25
    if c['guess',1] < c['reflen']: 
        b *= math.exp(1-c['reflen']/c['guess',1])
    return b

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('test_file', metavar='predict', help='predicted translations')
    argparser.add_argument('ref_file', metavar='true', help='true translations')
    argparser.add_argument('-n', help='maximum n-gram size to score', default=4, type=int)
    args = argparser.parse_args()

    test_sents = [line.split() for line in open(args.test_file)]
    ref_sents = [line.split() for line in open(args.ref_file)]

    print("BLEU:", score(test_sents, ref_sents, n=args.n))
    
