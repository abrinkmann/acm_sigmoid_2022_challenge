
import itertools
import logging
import re
from collections import defaultdict
from multiprocessing import Pool

import faiss
import torch

import numpy as np
from psutil import cpu_count
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

def block_with_bm25(X, attr, expected_cand_size, k_hits):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Preprocessing products...")

    worker = cpu_count()
    pool = Pool(worker)
    X['preprocessed'] = pool.map(preprocess_input, tqdm(list(X[attr].values)))
    pool.close()
    pool.join()

    pattern2id_1 = defaultdict(list)
    logger.info("Group products...")

    for i in tqdm(range(X.shape[0])):
        pattern2id_1[X['preprocessed'][i]].append(X['id'][i])

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

    logger.info("Load models...")


    model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

    def encode_and_embed(examples):
        # tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
        tokenized_output = tokenizer(examples, padding=True, truncation=True, max_length=16)
        encoded_output = model(input_ids=torch.tensor(tokenized_output['input_ids']),
                               attention_mask=torch.tensor(tokenized_output['attention_mask']),
                               token_type_ids=torch.tensor(tokenized_output['token_type_ids']))
        result = encoded_output['pooler_output'].detach().numpy()
        return result


    logger.info("Encode & Embed entities...")

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in tqdm(range(0, len(lst), n)):
            yield lst[i:i + n]

    # To-Do: Change embedding to list --> finally concatenate
    embeddings = []
    for examples in chunks(list(pattern2id_1.keys()), 256):
        embeddings.append(encode_and_embed(examples))

    embeddings = np.concatenate(embeddings)
    # # To-Do: Make sure that the embeddings are normalized
    logger.info('Add embeddings to faiss index')
    faiss_index = faiss.IndexFlatIP(256)
    faiss_index.add(embeddings)

    # logger.info("Search products...")
    # # # To-Do: Replace iteration
    # candidate_group_pairs = []
    # for index in tqdm(range(len(embeddings))):
    #     embedding = np.array([embeddings[index]])
    #     D, I = faiss_index.search(embedding, k_hits)
    #     for distance, top_id in zip(D[0], I[0]):
    #         if index == top_id:
    #             continue
    #         elif index < top_id:
    #             candidate_group_pair = (index, top_id)
    #         else:
    #             candidate_group_pair = (top_id, index)
    #
    #         candidate_group_pairs.append(candidate_group_pair)
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

def preprocess_input(doc):
    doc = doc[0].lower()

    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                  'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '[\d\w]*\.ca', '[\d\w]*\.fr', '[\d\w]*\.de',
                  '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    for stop_word in stop_words:
        doc = doc.replace(stop_word, ' ')

    for regex in regex_list:
        doc = re.sub(regex, ' ', doc)

    doc = re.sub('\s\s+', ' ', doc)
    doc = re.sub('\s*$', '', doc)
    doc = re.sub('^\s*', '', doc)

    tokens = tokenizer.tokenize(doc)
    pattern = tokenizer.convert_tokens_to_string(tokens[:16])

    return pattern

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
