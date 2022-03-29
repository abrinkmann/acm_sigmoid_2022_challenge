import itertools
import logging

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd



def block_with_bm25(X, attr):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Indexing products...")

    X['preprocessed'] = X.apply(lambda row: preprocess_input(row), axis=1)
    X_grouped = X.groupby(by=['preprocessed'])['id'].apply(list).reset_index(name='ids')
    #print(X_grouped.columns)
    #print(X_grouped['ids'].head())
    #print(len(X_grouped))

    X_grouped['tokenized'] = X_grouped.apply(lambda row: generate_tokenize_input(row['preprocessed']), axis=1)
    bm25 = BM25Okapi(X_grouped['tokenized'].values)
    k = 10

    logger.info("Searching products...")
    candidate_group_pairs = []
    #counter = 0
    for index, row in tqdm(X_grouped.iterrows()):
        doc_scores = bm25.get_scores(row['tokenized'])
        for top_id in np.argsort(doc_scores)[::-1][:k]:
            if index != top_id:
                normalized_score = doc_scores[top_id]/ np.amax(doc_scores)
                if normalized_score < 0.1:
                    break

                s1 = set(row['tokenized'])
                s2 = set(X_grouped.iloc[top_id]['tokenized'])
                jaccard_sim = len(s1.intersection(s2)) / max(len(s1), len(s2))
                if jaccard_sim < 0.15:
                    # Filter unlikely pairs with jaccard below or equal to 0.2
                    #counter += 1
                    continue

                if index < top_id:
                    candidate_group_pair = (index, top_id, normalized_score)
                elif index > top_id:
                    candidate_group_pair = (top_id, index, normalized_score)
                else:
                    continue

                candidate_group_pairs.append(candidate_group_pair)

    candidate_group_pair_dict = {}
    for candidate_group_pair in candidate_group_pairs:
        pair = (candidate_group_pair[0], candidate_group_pair[1])
        if pair in candidate_group_pair_dict:
            if candidate_group_pair_dict[pair] < candidate_group_pair[2]:
                candidate_group_pair_dict[pair] = candidate_group_pair[2]
        else:
            candidate_group_pair_dict[pair] = candidate_group_pair[2]
    #candidate_group_pairs = list(set(candidate_group_pairs))
    #candidate_pairs = determine_transitive_matches(list(candidate_pairs))

    # sort candidate pairs by normalized BM25 similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    normalized_similarities = []
    candidate_pairs_real_ids = []
    # Add candidates from grouping
    for candidate_list in X_grouped['ids']:
        if len(candidate_list) > 1:
            candidate_pairs = []
            for real_id1, real_id2 in itertools.permutations(candidate_list, 2):
                if real_id1 < real_id2:
                    candidate_pair = (real_id1, real_id2)
                elif real_id1 > real_id2:
                    candidate_pair = (real_id2, real_id1)
                else:
                    continue
                candidate_pairs.append(candidate_pair)

            candidate_pairs = list(set(candidate_pairs))
            for candidate_pair in candidate_pairs:
                candidate_pairs_real_ids.append(candidate_pair)
                normalized_similarities.append(1.0)


    # Add candidates from BM25 retrieval
    for pair, normalized_score in tqdm(candidate_group_pair_dict.items()):
        id1, id2 = pair

        # Determine real ids

        real_group_ids_1 = X_grouped['ids'][id1]
        real_group_ids_2 = X_grouped['ids'][id2]

        for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            candidate_pairs_real_ids.append(candidate_pair)
            normalized_similarities.append(normalized_score)

    candidate_pairs_real_ids = [x for _, x in sorted(zip(normalized_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids
    #return candidate_pairs_real_ids


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


def preprocess_input(row):
    # To-Do: Improve tokenizer
    doc = ' '.join(
        [str(value) for key, value in row.to_dict().items() if not (type(value) is float and np.isnan(value)) and key != 'id'])[:64]
    return doc


def generate_tokenize_input(preprocessed):
    tokenized = preprocessed.lower().split(' ')
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
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv")

    # perform blocking
    X1_candidate_pairs = block_with_bm25(X1, attr="title")
    X2_candidate_pairs = block_with_bm25(X2, attr="name")

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
