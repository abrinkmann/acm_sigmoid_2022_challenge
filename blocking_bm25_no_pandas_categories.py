import gc
import itertools
import logging
import os
import re
import time
from collections import defaultdict
from multiprocessing import Queue, Value, Process

import numpy as np
from multiprocess.pool import Pool
from psutil import cpu_count
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd



def block_with_bm25(X, attrs, expected_cand_size, k_hits, brands):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()
    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    logger.info("Preprocessing products...")
    docbrand2pattern2id = defaultdict(lambda: defaultdict(list))
    for i in tqdm(range(X.shape[0])):
        doc = ' '.join(
            [str(X[attr][i]) for attr in attrs if
             not (type(X[attr][i]) is float and np.isnan(X[attr][i]))]).lower()
        doc_brand = 'Null'
        for brand in brands:
            if brand in doc:
                doc_brand = brand
                break
        pattern_1 = preprocess_input(doc, stop_words, regex_list)
        docbrand2pattern2id[doc_brand][pattern_1].append(X['id'][i])

    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates')
    # Add candidates from grouping
    candidate_pairs_real_ids = []

    for doc_brand in tqdm(docbrand2pattern2id.keys()):
        for ids in docbrand2pattern2id[doc_brand].values():
            ids = list(sorted(ids))
            for j in range(len(ids)):
                for k in range(j + 1, len(ids)):
                    candidate_pairs_real_ids.append((ids[j], ids[k]))

    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))
    worker = cpu_count()
    pool = Pool(worker)

    new_candidate_pairs_real_ids = pool.starmap(search_per_doc_brand, zip(docbrand2pattern2id.values(),
                                                                          itertools.repeat(k_hits)))

    new_candidate_pairs_real_ids = list(set(itertools.chain(*new_candidate_pairs_real_ids)))
    candidate_pairs_real_ids.extend(new_candidate_pairs_real_ids)

    pool.close()
    pool.join()

    return candidate_pairs_real_ids


def tokenize(value):
    return value.split(' ')


def search_per_doc_brand(doc_brand, k_hits):
    new_candidate_pairs_real_ids = []
    goup_ids = [i for i in range(len(doc_brand.values()))]
    group2id_1 = dict(zip(goup_ids, doc_brand.values()))

    tokenized_corpus = [tokenize(value) for value in doc_brand.keys()]

    candidate_group_pairs = search_bm25(tokenized_corpus, k_hits)

    candidate_group_pairs = list(set(candidate_group_pairs))

    for pair in candidate_group_pairs:
        real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

        for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            new_candidate_pairs_real_ids.append(candidate_pair)

    return new_candidate_pairs_real_ids


def search_bm25(tokenized_corpus, k):
    # Index Corpus
    bm25 = BM25Okapi(tokenized_corpus)

    # Search Corpus
    candidate_group_pairs = []
    for index in range(len(tokenized_corpus)):
        if index < len(tokenized_corpus):
            query = tokenized_corpus[index]
            doc_scores = bm25.get_scores(query)
            for top_id in np.argsort(doc_scores)[::-1][:k]:
                if index != top_id:
                    normalized_score = doc_scores[top_id] / np.amax(doc_scores)
                    if normalized_score < 0.33:
                        break

                    if index < top_id:
                        candidate_group_pair = (index, top_id)
                    else:
                        candidate_group_pair = (top_id, index)

                    candidate_group_pairs.append(candidate_group_pair)

    #print('Result length: {}'.format(len(candidate_group_pairs)))
    return candidate_group_pairs


def preprocess_input(doc, stop_words, regex_list):
    # To-Do: Improve tokenizer
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

    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    stop_words_x1 = ['amazon.com', 'ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ', 'china']

    k_x_1 = 2
    brands_x_1 = ['vaio', 'samsung', 'fujitsu', 'lenovo', 'hp', 'hewlett-packard' 'asus', 'panasonic', 'toshiba',
                  'sony', 'aspire', 'dell']
    X1_candidate_pairs = block_with_bm25(X_1, ["title"], expected_cand_size_X1, k_x_1, brands_x_1)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    k_x_2 = 2
    brands_x_2 = ['lexar', 'kingston', 'samsung', 'sony', 'toshiba', 'sandisk', 'intenso', 'transcend']
    X2_candidate_pairs = block_with_bm25(X_2, ["name"], expected_cand_size_X1, k_x_2, brands_x_2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
