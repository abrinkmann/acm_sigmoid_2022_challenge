
import itertools
import logging
import re
from collections import defaultdict

import faiss
import torch

import numpy as np
from multiprocess.pool import Pool
from psutil import cpu_count
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel



def block_with_bm25(X, attrs, expected_cand_size, k_hits):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()
    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '-$', ':\s', '\(', '\)', ',']

    logger.info("Preprocessing products...")
    pattern2id_1 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        doc = ' '.join(
            [str(X[attr][i]) for attr in attrs if
             not (type(X[attr][i]) is float and np.isnan(X[attr][i]))]).lower()
        pattern_1 = preprocess_input(doc, stop_words, regex_list)
        pattern2id_1[pattern_1].append(X['id'][i])

    goup_ids = [i for i in range(len(pattern2id_1))]
    group2id_1 = dict(zip(goup_ids, pattern2id_1.values()))

    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates')
    # Add candidates from grouping
    candidate_pairs_real_ids = []

    for ids in tqdm(group2id_1.values()):
        ids = list(sorted(ids))
        for j in range(len(ids)):
            for k in range(j + 1, len(ids)):
                candidate_pairs_real_ids.append((ids[j], ids[k]))

    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))

    if len(candidate_pairs_real_ids) < expected_cand_size:
        logger.info("Load models...")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
        model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

        def encode_and_embed(examples):
            # tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
            tokenized_output = tokenizer(examples, padding=True, truncation=True, max_length=64)
            encoded_output = model(input_ids=torch.tensor(tokenized_output['input_ids']),
                                   attention_mask=torch.tensor(tokenized_output['attention_mask']),
                                   token_type_ids=torch.tensor(tokenized_output['token_type_ids']))
            result = encoded_output['pooler_output'].detach().numpy()
            return result

        from datasets import Dataset
        logger.info("Encode & Embed entities...")
        ds = Dataset.from_dict({'corpus': list(pattern2id_1.keys())})
        ds_with_embeddings = ds.map(lambda examples: {'embeddings': encode_and_embed(examples['corpus'])}, batched=True,
                                    batch_size=16, num_proc=cpu_count())
        ds_with_embeddings.add_faiss_index(column='embeddings')

        # worker = cpu_count()
        # pool = Pool(worker)
        # # Introduce batches(?)
        # embedded_corpus = pool.map(encode_and_embed, tqdm(list(pattern2id_1.keys())))
        # # To-Do: Make sure that the embeddings are normalized
        # faiss_index = faiss.IndexFlatIP(256)
        # for i in range(len(embedded_corpus)):
        #     faiss_index.add(embedded_corpus[i])

        logger.info("Search products...")
        # # To-Do: Replace iteration
        # candidate_group_pairs = []
        # for index in range(len(embedded_corpus)):
        #     D, I = faiss_index.search(embedded_corpus[index], k_hits)
        #     for i in range(0, len(I)):
        #         for distance, top_id in zip(D[i], I[i]):
        #             if index == top_id:
        #                 continue
        #             elif index < top_id:
        #                 candidate_group_pair = (index, top_id)
        #             else:
        #                 candidate_group_pair = (top_id, index)
        #
        #             candidate_group_pairs.append(candidate_group_pair)
        #
        # candidate_group_pairs = list(set(candidate_group_pairs))
        # if len(candidate_group_pairs) > (expected_cand_size - len(candidate_pairs_real_ids) + 1):
        #     candidate_group_pairs = candidate_group_pairs[:(expected_cand_size - len(candidate_pairs_real_ids) + 1)]
        #
        # logger.info('GroupIds to real ids')
        # for pair in tqdm(candidate_group_pairs):
        #     real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        #     real_group_ids_2 = list(sorted(group2id_1[pair[1]]))
        #
        #     for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
        #         if real_id1 < real_id2:
        #             candidate_pair = (real_id1, real_id2)
        #         elif real_id1 > real_id2:
        #             candidate_pair = (real_id2, real_id1)
        #         else:
        #             continue
        #         candidate_pairs_real_ids.append(candidate_pair)

    return candidate_pairs_real_ids


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
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']

    k_x_1 = 2
    X1_candidate_pairs = block_with_bm25(X_1, ["title"], expected_cand_size_X1, k_x_1)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    k_x_2 = 2
    X2_candidate_pairs = block_with_bm25(X_2, ["name"], expected_cand_size_X1, k_x_2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
