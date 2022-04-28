import csv
from itertools import repeat, product
import logging
import math
import re
from collections import defaultdict
from multiprocessing import Pool

import faiss
import numpy as np
import torch

from psutil import cpu_count
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

#from blocking import block_with_attr
from blocking import block_with_attr
from model_contrastive import ContrastivePretrainModel


def load_normalization():
    """Load Normalization file - Especially for D2"""
    normalizations = {}
    with open('normalization.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_values = line.split(',')
            normalizations[line_values[0]] = line_values[1].replace('\n','')

    return normalizations

#normalizations = {}


def block_regex(X, attr, path_to_preprocessed_file, norm, char_length):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Preprocessing products...")

    worker = cpu_count()
    pool = Pool(worker)
    X['preprocessed'] = pool.starmap(preprocess_input, zip(list(X[attr].values), repeat(norm), repeat(char_length)))

    if path_to_preprocessed_file is not None:
        X['tokens'] = pool.map(tokenize_input, tqdm(list(X['preprocessed'].values)))
        X.to_csv(path_to_preprocessed_file, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_MINIMAL)

    pool.close()
    pool.join()

    logger.info("Group products...")
    pattern2id_1 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        pattern2id_1[X['preprocessed'][i]].append(X['id'][i])

    # Prepare pairs deduced from groups while waiting for search results
    # To-DO: Parallel processing of group candidate creation & model loading
    logger.info('Create group candidates')

    goup_ids = [i for i in range(len(pattern2id_1))]
    group2id_1 = dict(zip(goup_ids, pattern2id_1.values()))

    # Create first candidates in subprocess while models are loaded
    candidate_pairs_real_ids = []
    for ids in tqdm(group2id_1.values()):
        ids = list(sorted(ids))
        for j in range(len(ids)):
            for k in range(j + 1, len(ids)):
                candidate_pairs_real_ids.append((ids[j], ids[k]))

    regex_pattern2id_2 = defaultdict(list)
    for i in range(len(pattern2id_1)):
        pattern = list(pattern2id_1.keys())[i]
        regex_pattern_2 = re.findall("\w+\s\w+\d+", pattern)  # look for patterns like "thinkpad x1"
        for regex_match in regex_pattern_2:
            regex_pattern2id_2[regex_match].append(i)


    pair2sim = defaultdict(float)
    for i in range(len(regex_pattern2id_2)):
        matched_groups = sorted(list(regex_pattern2id_2.values())[i])
        for j in range(len(matched_groups)):
            for k in range(j+ 1, len(matched_groups)):
                group_1 = matched_groups[j]
                group_2 = matched_groups[k]

                tokens_1 = set(list(pattern2id_1.keys())[group_1].split(' '))
                tokens_2 = set(list(pattern2id_1.keys())[group_2].split(' '))

                if group_1 < group_2:
                    candidate_pair = (group_1, group_2)
                elif group_1 > group_2:
                    candidate_pair = (group_1, group_2)
                else:
                    continue

                pair2sim[candidate_pair] = len(tokens_1.intersection(tokens_2)) / max(len(tokens_1), len(tokens_2))



    candidate_group_pairs = [k for k, _ in sorted(pair2sim.items(), key=lambda k_v: k_v[1], reverse=True)]

    logger.info('GroupIds to real ids')
    for pair in tqdm(candidate_group_pairs):
        real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

        #cluster_size = 0
        for real_id1, real_id2 in product(real_group_ids_1, real_group_ids_2):
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            candidate_pairs_real_ids.append(candidate_pair)

            # if cluster_size_threshold is not None:
            #     cluster_size += 1
            #     if cluster_size >= cluster_size_threshold:
            #         break

    return candidate_pairs_real_ids


def preprocess_input(docs, normalizations, char_length):
    if len(docs) == 0:
        return ''
    else:
        doc = ' '.join([str(value) for value in docs if type(value) is str or (type(value) is float and not np.isnan(value))]).lower()

        stop_words = ['ebay', 'google', 'vology', 'buy', 'cheapest', 'foto de angelis', 'cheap', 'core',
                      'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '\'\'', '"', '\\\\n',
                      'tesco direct', 'color', ' y ', ' et ', 'tipo a', 'type-a', 'type a', 'informática', ' de ',
                      ' con ', 'newest', ' new', ' ram ', '64-bit', '32-bit', 'accessories', 'series', 'touchscreen',
                      'product', 'customized']

        stop_signs = ['&nbsp;', '&quot;', '&amp;', ',', ';', '-', ':', '|', '/', '(', ')', '/', '&']

        regex_list_1 = ['^dell*', '[\d\w]*\.com', '[\d\w]*\.ca', '[\d\w]*\.fr', '[\d\w]*\.de', '[\d\w]*\.es',
                        '(\d+\s*gb\s*hdd|\d+\s*gb\s*ssd)', '\\\\n']

        for stop_word in stop_words:
            doc = doc.replace(stop_word, ' ')

        for stop_sign in stop_signs:
            doc = doc.replace(stop_sign, ' ')

        for regex in regex_list_1:
            doc = re.sub(regex, '', doc)

        # Move GB pattern to beginning of doc
        gb_pattern = re.findall('(d+\s*gbbeuk|\d+\s*gbbeu|\d+\s*gb|\d+\s*go|\d+\s*bbeu|\d+\s*gabeu)', doc)

        if len(gb_pattern) > 0:
            gb_pattern.sort()
            while len(gb_pattern) > 0 and gb_pattern[0][0] == '0':
                gb_pattern.remove(gb_pattern[0])

            if len(gb_pattern) > 0:
                doc = re.sub('(d+\s*gbbeuk|\d+\s*gbbeu|\d+\s*gb|\d+\s*go|\d+\s*bbeu|\d+\s*gabeu)', ' ', doc)
                doc = '{} {}'.format(gb_pattern[0].replace(' ', '').replace('go', 'gb').replace('gbbeuk', 'gb').replace('gbbeu', 'gb').replace('bbeu', 'gb'),
                                     doc)  # Only take the first found pattern --> might lead to problems, but we need to focus on the first tokens.

        doc = re.sub('\s\s+', ' ', doc)

        if normalizations is not None:
            for key in normalizations:
                doc = doc.replace(key, normalizations[key])
            doc = re.sub('\s\s+', ' ', doc)
            # Clean up normalization
            doc = doc.replace('usb stick usb stick', 'usb stick')
            doc = doc.replace('usb stick usb', 'usb stick')
            doc = doc.replace('usb usb', 'usb')
            doc = doc.replace('memory card memory card', 'memory card')
            doc = doc.replace('memory card memory', 'memory card')
            doc = doc.replace('memory memory', 'memory')
            doc = doc.replace('card card', 'card')
            doc = doc.replace('windows windows', 'windows')
            doc = doc.replace('laptop laptop', 'laptop')
            doc = doc.replace('hp hp', 'hp')

        doc = re.sub('\s\s+', ' ', doc)
        doc = re.sub('\s*$', '', doc)
        doc = re.sub('^\s*', '', doc)

        return doc[:char_length]


def determine_transitive_matches(pairs2sim):

    candidate_group_pairs = sorted(list(pairs2sim.keys()))

    for j in tqdm(range(len(candidate_group_pairs))):
        for k in range(j + 1, len(candidate_group_pairs)):
            first_group = candidate_group_pairs[j]
            second_group = candidate_group_pairs[k]
            if first_group[1] > second_group[0]:
                break

            matches = sum([1 for element in second_group if element in first_group])
            if matches > 0:
                group_1 = [element for element in second_group if element not in first_group][0]
                group_2 = [element for element in first_group if element not in second_group][0]
                if group_1 < group_2:
                    potential_match = (group_1, group_2)
                else:
                    potential_match = (group_2, group_1)

                match_sim = sum([pairs2sim[first_group], pairs2sim[second_group]]) / 2
                pairs2sim[potential_match] = max(pairs2sim[potential_match], match_sim)

    return pairs2sim


def tokenize_input(doc):
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
    logger = logging.getLogger()

    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # Local Testing - COMMENT FOR SUBMISSION!
    logger.warning('NOT A REAL SUBMISSION!')
    expected_cand_size_X1 = 2814
    expected_cand_size_X2 = 4392

    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    k_x_1 = 30
    seq_length_x_1 = 32
    proj_x_1 = 32
    normalizations_x_1 = load_normalization()
    #cluster_size_threshold_x1 = None
    transitive_closure_x_1 = False
    X1_candidate_pairs = block_regex(X_1, ["title"], None, normalizations_x_1, 120)
    #X1_candidate_pairs = block_with_attr(X_1, "title")
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    k_x_2 = 30
    seq_length_x_2 = 24
    proj_x_2 = 16
    normalizations_x_2 = normalizations_x_1
    #cluster_size_threshold_x2 = None
    transitive_closure_x_2 = False
    X2_candidate_pairs = block_regex(X_2, ["name"], None, normalizations_x_2, 120)
    # X2_candidate_pairs = block_neural(X_2, ["name"], k_x_2, None, normalizations_x_2, 'supcon',
    #                                   'models/supcon/len{}/X2_model_len{}_trans{}.bin'.format(seq_length_x_2, seq_length_x_2,
    #                                                                                           proj_x_2), seq_length_x_2, proj_x_2, transitive_closure_x_2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)

    if expected_cand_size_X1 < 1000000:
        logger.warning('NOT A REAL SUBMISSION!')
    logger.info('Candidates saved!')
