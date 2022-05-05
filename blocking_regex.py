import logging
import re
from collections import defaultdict

from tqdm import tqdm
import pandas as pd


def block_with_regex(X, attr, path_to_preprocessed_file):  # replace with your logic.
    '''
        This function performs blocking using attr
        :param X: dataframe
        :param attr: attribute used for blocking
        :return: candidate set of tuple pairs
        '''

    # build index from patterns to tuples
    pattern2id_2 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i]).lower()

        tokens = re.split('[^a-zA-Z0-9]', attr_i)
        for token in tokens:
            if len(token) > 7 and any([ch.isdigit() for ch in token]):
                pattern2id_2[token].append(i)


    # add id pairs that share the same pattern to candidate set
    candidate_pairs_2 = []
    for pattern in tqdm(pattern2id_2):
        print(pattern)
        ids = list(sorted(pattern2id_2[pattern]))
        if len(ids) < 50:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_2.append((ids[i], ids[j]))

    # remove duplicate pairs and take union
    candidate_pairs = list(set(candidate_pairs_2))

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get real ids
        real_id1 = X['id'][id1]
        real_id2 = X['id'][id2]
        if real_id1 < real_id2:  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # compute jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]

    return candidate_pairs_real_ids


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
    logger = logging.getLogger()

    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # Local Testing - COMMENT FOR SUBMISSION!
    logger.warning('NOT A REAL SUBMISSION!')
    expected_cand_size_X1 = 2814
    expected_cand_size_X2 = 4392


    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    X1_candidate_pairs = block_with_regex(X_1, "title", None)

    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    X2_candidate_pairs = block_with_regex(X_2, "name", None)

    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)

    if expected_cand_size_X1 < 1000000:
        logger.warning('NOT A REAL SUBMISSION!')
    logger.info('Candidates saved!')
