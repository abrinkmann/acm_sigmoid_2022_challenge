import itertools
import logging
import re
import time
from collections import defaultdict

import gensim
from multiprocessing import Pool, Queue, Process

import numpy as np
from gensim import corpora
from psutil import cpu_count
from tqdm import tqdm
import pandas as pd
from gensim.models import TfidfModel


def preprocess_X(X, attr):
    logger = logging.getLogger()

    logger.info("Preprocessing products...")
    worker = cpu_count()
    pool = Pool(worker)
    X['preprocessed'] = pool.map(preprocess_input, tqdm(list(X[attr].values)))
    pool.close()
    pool.join()

    return X


def block_with_tf_idf(X, k_hits):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''
    logger = logging.getLogger()

    pattern2id = defaultdict(list)
    logger.info("Group products...")
    for i in tqdm(range(X.shape[0])):
        pattern = X['preprocessed'][i]
        pattern2id[pattern].append(X['id'][i])

    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates')
    # Add candidates from grouping
    candidate_pairs_real_ids = []

    for ids in tqdm(pattern2id.values()):
        ids = list(sorted(ids))
        for j in range(len(ids)):
            for k in range(j + 1, len(ids)):
                candidate_pairs_real_ids.append((ids[j], ids[k]))

    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))
    jaccard_similarities = [1.0] * len(candidate_pairs_real_ids)

    logger.info('Start search')

    new_candidate_pairs_real_ids, new_jaccard_similarities = search_tfidf_gensim(pattern2id, k_hits)
    candidate_pairs_real_ids.extend(new_candidate_pairs_real_ids)
    jaccard_similarities.extend(new_jaccard_similarities)

    candidate_pairs_real_ids = [x for _, x in
                                sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]

    logger.info('Finished search')

    return candidate_pairs_real_ids


def tokenize(value):
    return value.split(' ')


def tokenize_tfidf_vectorizer(value):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokens = token_pattern.findall(value)

    return tokens

def search_tfidf_gensim(pattern2id, k_hits):

    logger = logging.getLogger()
    candidate_group_pairs = []
    tokenized_corpus = [tokenize(product) for product in pattern2id.keys()]
    dct = corpora.Dictionary(tokenized_corpus)

    if len(tokenized_corpus) * 0.5 > 2: # fit dictionary
        dct.filter_extremes(no_below=2, no_above=0.8)

    corpus = [dct.doc2bow(product) for product in tokenized_corpus]

    tfidf = TfidfModel(corpus)
    #corpus_tfidf = tfidf[corpus]
    logger.info('Create Similarity Matrix')
    index = gensim.similarities.Similarity(output_prefix=None, corpus=tfidf[corpus], num_features=len(dct),
                                           num_best=k_hits)

    logger.info('Collect similarities')
    i = 0
    for sims in index:
        for hit in sims:
            top_id, _ = hit
            if i != top_id:
                if i < top_id:
                    candidate_group_pair = (i, top_id)
                else:
                    candidate_group_pair = (top_id, i)

                candidate_group_pairs.append(candidate_group_pair)
        i += 1

    candidate_group_pairs = list(set(candidate_group_pairs))
    new_candidate_pairs_real_ids = []
    new_jaccard_similarities = []
    goup_ids = [i for i in range(len(pattern2id.values()))]
    group2id_1 = dict(zip(goup_ids, pattern2id.values()))

    for pair in candidate_group_pairs:
        real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

        s1 = set(tokenized_corpus[pair[0]])
        s2 = set(tokenized_corpus[pair[1]])
        jaccard_similarity = len(s1.intersection(s2)) / max(len(s1), len(s2))

        if jaccard_similarity > 0.2:
            for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
                if real_id1 < real_id2:
                    candidate_pair = (real_id1, real_id2)
                elif real_id1 > real_id2:
                    candidate_pair = (real_id2, real_id1)
                else:
                    continue
                new_candidate_pairs_real_ids.append(candidate_pair)
                new_jaccard_similarities.append(jaccard_similarity)

    return (new_candidate_pairs_real_ids, new_jaccard_similarities)

def preprocess_input(doc):
    doc = doc.lower()

    stop_words = ['ebay', 'google', 'vology', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    doc = doc.replace('hewlett-packard', 'hp')

    for stop_word in stop_words:
        doc = doc.replace(stop_word, ' ')

    for regex in regex_list:
        doc = re.sub(regex, ' ', doc)

    doc = re.sub('\s\s+', ' ', doc)
    doc = re.sub('\s*$', '', doc)
    doc = re.sub('^\s*', '', doc)

    return doc[:64]


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    X_1 = pd.read_csv("../X1.csv")
    X_2 = pd.read_csv("../X2.csv")

    stop_words_x1 = ['amazon.com', 'ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ', 'china']

    k_x_1 = 25
    X_1_preprocessed = preprocess_X(X_1, "title")

    k_x_2 = 25
    X_2_preprocessed = preprocess_X(X_2, "name")

    worker = cpu_count()
    pool = Pool(worker)
    candidate_pairs = pool.starmap(block_with_tf_idf, zip([X_1_preprocessed, X_2_preprocessed], [k_x_1, k_x_2]))
    pool.close()
    pool.join()


    X1_candidate_pairs = candidate_pairs[0]
    X2_candidate_pairs = candidate_pairs[1]


    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
