import itertools
import logging
import os
import re
import time

import numpy as np
from multiprocess.pool import Pool
from psutil import cpu_count
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd



def block_with_bm25(path_to_X, attr, stop_words, normalization):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Load dataset...")
    X = pd.read_csv(path_to_X)

    logger.info("Indexing products...")
    X['preprocessed'] = X.apply(lambda row: preprocess_input(row, stop_words, normalization), axis=1)
    X.to_csv(path_to_X.replace('.csv', '_preprocessed.csv'))
    # Introduce multiprocessing!
    X_grouped = X.groupby(by=['preprocessed'])['id'].apply(list).reset_index(name='ids')
    #print(X_grouped.columns)
    #print(X_grouped['ids'].head())
    #print(len(X_grouped))

    X_grouped['tokenized'] = X_grouped.apply(lambda row: generate_tokenized_input(row['preprocessed']), axis=1)
    k = 2

    logger.info("Searching products...")
    candidate_group_pairs = []
    worker = os.environ['WORKER']
    logger.info('Number of initialized workers {}'.format(worker))
    pool = Pool(worker)
    results = []

    multiprocessing_step_size = int((X_grouped.shape[0] / worker)) + 1
    start = 0
    logger.info('Start search!')
    while start < X_grouped.shape[0]:
        value_range = (start, start+multiprocessing_step_size)
        results.append(
            pool.apply_async(search_bm25, (value_range, X_grouped, k,)))

        start += multiprocessing_step_size

    # Prepare pairs deduced from groups while waiting for search results
    candidate_pairs_real_ids = []
    # Add candidates from grouping
    for i in tqdm(range(X_grouped.shape[0])):
        ids = list(sorted(X_grouped['ids'][i]))
        if len(ids) > 1:
            for j in range(len(ids)):
                for k in range(j + 1, len(ids)):
                    candidate_pairs_real_ids.append((ids[j], ids[k]))

    jaccard_similarities = [1.0] * len(candidate_pairs_real_ids)

    # Collect search results
    logger.info('Collect search results!')
    while len(results) > 0:
        collected_results = []
        for result in results:
            if result.ready():
                candidate_group_pair = result.get()
                candidate_group_pairs.extend(candidate_group_pair)
                collected_results.append(result)

        results = [result for result in results if result not in collected_results]

    candidate_group_pairs = list(set(candidate_group_pairs))
    #candidate_group_pairs = determine_transitive_matches(list(candidate_group_pairs))
    #candidate_pairs = determine_transitive_matches(list(candidate_pairs))

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs

    # Add candidates from BM25 retrieval
    for pair in tqdm(candidate_group_pairs):
        id1, id2 = pair

        # Determine real ids

        real_group_ids_1 = list(sorted(X_grouped['ids'][id1]))
        real_group_ids_2 = list(sorted(X_grouped['ids'][id2]))

        for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            candidate_pairs_real_ids.append(candidate_pair)

            # Calculate jaccard similarity
            s1 = set(X_grouped['tokenized'][id1])
            s2 = set(X_grouped['tokenized'][id2])
            jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))

    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids
    #return candidate_pairs_real_ids


def search_bm25(value_range, X_grouped, k):
    bm25 = BM25Okapi(X_grouped['tokenized'].values)
    candidate_group_pairs = []

    for index in range(value_range[0], value_range[1]):
        if index < X_grouped.shape[0]:
            doc_scores = bm25.get_scores(X_grouped['tokenized'][index])
            for top_id in np.argsort(doc_scores)[::-1][:k]:
                if index != top_id:
                    normalized_score = doc_scores[top_id] / np.amax(doc_scores)
                    if normalized_score < 0.33:
                        break

                    if index < top_id:
                        candidate_group_pair = (index, top_id)
                    elif index > top_id:
                        candidate_group_pair = (top_id, index)
                    else:
                        continue

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


def preprocess_input(row, stop_words, normalization):
    # To-Do: Improve tokenizer
    doc = ' '.join(
        [str(value).lower() for key, value in row.to_dict().items() if not (type(value) is float and np.isnan(value)) ])
         #and key != 'id'])
    regex_list = ['[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '-$', ':\s', '\(', '\)', ',']
    for regex in regex_list:
        doc = re.sub(regex, ' ', doc)

    for stop_word in stop_words:
        doc = doc.replace(stop_word, ' ')

    for not_normalized_value in normalization:
        if not_normalized_value in doc:
            doc = doc.replace(not_normalized_value, normalization[not_normalized_value])

    doc = re.sub('\s\s+', ' ', doc)
    doc = re.sub('\s*$', '', doc)
    doc = re.sub('^\s*', '', doc)
    doc = doc[:64]

    return doc


def generate_tokenized_input(preprocessed):
    tokenized = preprocessed.split(' ')
    return tokenized


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

    stop_words_x1 = ['amazon.com', 'ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    normalization_X1 = {}
    X1_candidate_pairs = block_with_bm25("X1_extended.csv", "title", stop_words_x1, normalization_X1)
    stop_words_x2 = []
    normalization_X2 = {}
    X2_candidate_pairs = block_with_bm25("X2_extended.csv", "name", stop_words_x2, normalization_X2)

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
