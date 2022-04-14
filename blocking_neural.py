import csv
import itertools
import logging
import math
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool, Queue, Process

import faiss
import torch

from onnxruntime import InferenceSession
from psutil import cpu_count
from sentence_transformers import models, SentenceTransformer
from torch import nn
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')
model = SentenceTransformer('ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')

def block_neural(X, attr, k_hits, path_to_preprocessed_file):  # replace with your logic.
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

    if path_to_preprocessed_file is not None:
        X['tokens'] = pool.map(tokenize_input, tqdm(list(X['preprocessed'].values)))
        X.to_csv(path_to_preprocessed_file, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_MINIMAL)

    pool.close()
    pool.join()

    pattern2id_1 = defaultdict(list)
    logger.info("Group products...")

    for i in tqdm(range(X.shape[0])):
        pattern2id_1[X['preprocessed'][i]].append(X['id'][i])

    # Prepare pairs deduced from groups while waiting for search results
    # To-DO: Parallel processing of group candidate creation & model loading
    logger.info('Create group candidates')
    goup_ids = [i for i in range(len(pattern2id_1))]
    group2id_1 = dict(zip(goup_ids, pattern2id_1.values()))
    # Add candidates from grouping
    candidate_pairs_real_ids = []

    for ids in tqdm(group2id_1.values()):
        ids = list(sorted(ids))
        for j in range(len(ids)):
            for k in range(j + 1, len(ids)):
                candidate_pairs_real_ids.append((ids[j], ids[k]))

    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))

    logger.info("Encode & Embed entities...")

    onnx_run = True
    if onnx_run:
        # session = InferenceSession("embeddings.onnx")
        #
        # tokens = tokenizer(list(pattern2id_1.keys()), return_tensors="np")
        # embeddings = session.run(None, dict(tokens))[0]
        
        session = InferenceSession("embeddings.onnx", providers=['CPUExecutionProvider'])
        tokens = tokenizer(["I am happy", "I am glad"], return_tensors="np")
        embeddings = session.run(None, dict(tokens))[0]

    else:
        embeddings = model.encode(list(pattern2id_1.keys()), batch_size=256, show_progress_bar=True,
                                   normalize_embeddings=True)


    #embeddings = np.concatenate(embeddings)
    # # To-Do: Make sure that the embeddings are normalized
    logger.info('Initialize faiss index')
    d = 256
    ### m = 8
    nlist = int(20*math.sqrt(len(embeddings)))
    quantizer = faiss.IndexFlatIP(d)
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    #faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) # 8 specifies that each sub-vector is encoded as 8 bits

    assert not faiss_index.is_trained
    logger.info('Train Faiss Index')
    faiss_index.train(embeddings)
    assert faiss_index.is_trained
    logger.info('Add embeddings to faiss index')
    faiss_index.add(embeddings)

    logger.info("Search products...")
    candidate_group_pairs = []
    faiss_index.nprobe = 2     # the number of cells (out of nlist) that are visited to perform a search

    # for index in tqdm(range(len(embeddings))):
    #     embedding = np.array([embeddings[index]])

    D, I = faiss_index.search(embeddings, k_hits)
    logger.info('Collect search results')
    for index in tqdm(range(len(I))):
        for distance, top_id in zip(D[index], I[index]):
            if top_id > 0:
                if index == top_id:
                    continue
                elif index < top_id:
                    candidate_group_pair = (index, top_id, distance)
                else:
                    candidate_group_pair = (top_id, index, distance)

                candidate_group_pairs.append(candidate_group_pair)

    candidate_group_pairs = list(set(candidate_group_pairs))

    logger.info('GroupIds to real ids')
    for pair in tqdm(candidate_group_pairs):
        real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

        for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            candidate_pairs_real_ids.append(candidate_pair)

    return candidate_pairs_real_ids

# def encode_and_embed_parallel(input_q, output_q, num_threads):
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
#     special_tokens_dict = {
#         'additional_special_tokens': ['lenovo', 'thinkpad', 'elitebook', 'toshiba', 'asus', 'acer', 'lexar', 'sandisk',
#                                       'tesco', 'intenso', 'transcend']}
#     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#     model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
#     model.resize_token_embeddings(len(tokenizer))
#
#     ff_layer = nn.Linear(256, 64, bias=False)
#
#     torch.set_num_threads(num_threads)
#
#     while not input_q.empty():
#         examples = input_q.get()
#         with torch.no_grad():
#             tokenized_input = tokenizer(examples, padding=True, truncation=True, max_length=16, return_tensors='pt')
#             encoded_output = model(input_ids=tokenized_input['input_ids'],
#                                    attention_mask=tokenized_input['attention_mask'],
#                                    token_type_ids=tokenized_input['token_type_ids'])
#             #result = mean_pooling(encoded_output, tokenized_input['attention_mask'])
#             #result = ff_layer(result)
#
#         # tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
#         output_q.put(result)


def preprocess_input(doc):
    doc = doc[0].lower()

    stop_words = ['ebay', 'google', 'vology', 'buy', 'cheapest', 'cheap', 'core',
                  'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '&nbsp;', '& ', '', ';']
    regex_list_1 = ['^dell*', '[\d\w]*\.com', '[\d\w]*\.ca', '[\d\w]*\.fr', '[\d\w]*\.de', '(\d+\s*gb\s*hdd|\d+\s*gb\s*ssd)']

    regex_list_2 = ['\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    for stop_word in stop_words:
        doc = doc.replace(stop_word, '')

    for regex in regex_list_1:
        doc = re.sub(regex, '', doc)

    # Move GB pattern to beginning of doc
    gb_pattern = re.findall('(\d+\s*gb|\d+\s*go)', doc)

    if len(gb_pattern) > 0:
        gb_pattern.sort()
        for pattern in gb_pattern:
            doc = doc.replace(pattern, '')
        doc = '{} {}'.format(gb_pattern[0].replace(' ', ''), doc) # Only take the first found pattern --> might lead to problems, but we need to focus on the first 16 tokens.

    for regex in regex_list_2:
        doc = re.sub(regex, '', doc)

    doc = re.sub('\s\s+', ' ', doc)
    doc = re.sub('\s*$', '', doc)
    doc = re.sub('^\s*', '', doc)

    tokens = tokenizer.tokenize(doc)
    pattern = tokenizer.convert_tokens_to_string(tokens[:16])

    return pattern


def tokenize_input(doc):
    return tokenizer.tokenize(doc)


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

    k_x_1 = 3
    X1_candidate_pairs = block_neural(X_1, ["title"], k_x_1, None)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    k_x_2 = 3
    X2_candidate_pairs = block_neural(X_2, ["name"], k_x_2, None)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
