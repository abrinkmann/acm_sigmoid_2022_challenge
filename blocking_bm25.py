import gc
import itertools
import logging
import os
import re
import time
from multiprocessing import Queue, Value, Process

import numpy as np
from multiprocess.pool import Pool
from psutil import cpu_count
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd



def block_with_bm25(X, attr, expected_cand_size, path_to_X):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Preprocessing products...")
    X = parallel_processing(X, preprocess_dataframe)
    X.to_csv(path_to_X.replace('.csv', '_preprocessed.csv'))
    # Introduce multiprocessing!
    X_grouped = X.groupby(by=['preprocessed'])['id'].apply(list).reset_index(name='ids')
    #X_grouped = parallel_processing(X_grouped,count_ids_dataframe)
    #X_grouped = X_grouped.sort_values(by=['len_ids'])

    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates')
    # Add candidates from grouping
    worker = cpu_count()
    pool = Pool(worker)
    candidate_pairs_real_ids = pool.map(determine_pairs_by_group_ids, tqdm(X_grouped['ids'].values))
    pool.close()
    pool.join()

    candidate_pairs_real_ids = list(set(itertools.chain(*candidate_pairs_real_ids)))

    if len(candidate_pairs_real_ids) < expected_cand_size:
        logger.info("Tokenize products...")
        X_grouped = parallel_processing(X_grouped, tokenize_dataframe)
        k = 2

        #processes = []
        #for i in range(worker/2):
        logger.info('Index tokens!')
        bm25 = BM25Okapi(X_grouped['tokenized'].values)
        pool = Pool(worker)
        logger.info('Start search')
        candidate_group_pairs = pool.starmap(search_bm25, zip(itertools.repeat(bm25), X_grouped['tokenized'].values,
                                                              X_grouped.index, itertools.repeat(k)))

        candidate_group_pairs = list(set(itertools.chain(*candidate_group_pairs)))
        logger.info('Derive pairs from groups')
        if len(candidate_group_pairs) > expected_cand_size:
            candidate_group_pairs = candidate_group_pairs[:expected_cand_size]

        candidate_group_pair_ids = [(X_grouped['ids'][pair[0]], X_grouped['ids'][pair[1]]) for pair in candidate_group_pairs]
        new_candidate_pairs_real_ids = pool.map(determine_pairs_by_group_pairs, tqdm(candidate_group_pair_ids))
        candidate_pairs_real_ids.extend(list(set(itertools.chain(*new_candidate_pairs_real_ids))))

        pool.close()
        pool.join()
        #p.join()
        #p.close()
        # Determine transitive groups
        #if len(candidate_pairs_real_ids) < expected_cand_size:
        #    candidate_pairs_real_ids = determine_transitive_matches(candidate_pairs_real_ids)

    return candidate_pairs_real_ids


def search_bm25(bm25, query, index, k):

    candidate_group_pairs = []
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

    return candidate_group_pairs


def determine_transitive_matches(candidate_pairs):
    change = True
    while change:
        cluster = []
        #print(len(pairs))
        old_length = len(candidate_pairs)
        while len(candidate_pairs) > 0:
            pair = candidate_pairs.pop()
            removable_pairs = []
            for new_candidate_pair in candidate_pairs:
                matches = sum([1 for element in new_candidate_pair if element in pair])
                if matches > 0:
                    removable_pairs.append(new_candidate_pair)
                    pair = tuple(set(pair + new_candidate_pair))

            cluster.append(pair)
            candidate_pairs = [pair for pair in candidate_pairs if pair not in removable_pairs]

        candidate_pairs = cluster
        change = len(candidate_pairs) != old_length

    cluster_pair_dict = {'clu_{}'.format(i): candidate_pairs[i] for i in range(0, len(candidate_pairs))}
    new_candidate_pairs = []
    for pattern in tqdm(cluster_pair_dict):
        ids = list(sorted(cluster_pair_dict[pattern]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                new_candidate_pairs.append((ids[i], ids[j]))

    return new_candidate_pairs


def parallel_processing(X, func):
    worker = cpu_count()
    pool = Pool(worker)
    X_split = np.array_split(X, worker)
    X_split = pool.map(func, tqdm(X_split))
    pool.close()
    pool.join()

    return pd.concat(X_split)


def count_ids_dataframe(X):
    return X.apply(count_ids, axis=1)


def count_ids(row):
    row['len_ids'] = len(row['ids'])
    return row


def preprocess_dataframe(X):
    return X.apply(preprocess_input, axis=1)


def preprocess_input(row):
    stop_words = ['amazon.com', 'ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ', 'shop']
    # To-Do: Improve tokenizer
    doc = ' '.join(
        [str(value).lower() for key, value in row.to_dict().items() if not (type(value) is float and np.isnan(value))
         and key != 'id'])
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '^"', '-$', ':\s', '\(', '\)', ',', '\\n']
    for regex in regex_list:
        doc = re.sub(regex, ' ', doc)

    for stop_word in stop_words:
        doc = doc.replace(stop_word, ' ')

    doc = re.sub('\s\s+', ' ', doc)
    doc = re.sub('\s*$', '', doc)
    doc = re.sub('^\s*', '', doc)
    row['preprocessed'] = doc[:64]

    return row


def tokenize_dataframe(X):
    return X.apply(generate_tokenized_input, axis=1)


def generate_tokenized_input(row):
    row['tokenized'] = row['preprocessed'].split(' ')
    return row


def determine_pairs_by_group_ids(ids):
    candidate_pairs_real_ids = []
    # Add candidates from grouping
    ids = list(sorted(ids))
    for j in range(len(ids)):
        for k in range(j + 1, len(ids)):
            candidate_pairs_real_ids.append((ids[j], ids[k]))

    return candidate_pairs_real_ids


def determine_pairs_by_group_pairs(id_pair):
    # Determine real ids
    ids1, ids2 = id_pair
    real_group_ids_1 = list(sorted(ids1))
    real_group_ids_2 = list(sorted(ids2))

    new_candidate_pairs_real_ids = []
    for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
        if real_id1 < real_id2:
            candidate_pair = (real_id1, real_id2)
        elif real_id1 > real_id2:
            candidate_pair = (real_id2, real_id1)
        else:
            continue
        new_candidate_pairs_real_ids.append(candidate_pair)

    return list(set(new_candidate_pairs_real_ids))

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
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']

    X1_candidate_pairs = block_with_bm25(X_1, "title", expected_cand_size_X1, "../X1.csv")
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    X2_candidate_pairs = block_with_bm25(X_2, "name", expected_cand_size_X1, "../X2.csv")
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
