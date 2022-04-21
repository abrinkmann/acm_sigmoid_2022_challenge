import csv
import itertools
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

from model_contrastive import ContrastivePretrainModel

tokenizer = AutoTokenizer.from_pretrained('models/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')

def block_neural(X, attr, k_hits, path_to_preprocessed_file, model_type, model_path, proj):  # replace with your logic.
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

    # if path_to_preprocessed_file is not None:
    #     X['tokens'] = pool.map(tokenize_input, tqdm(list(X['preprocessed'].values)))
    #     X.to_csv(path_to_preprocessed_file, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_MINIMAL)

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

    if model_type == 'sbert':
        logger.info('Load Models')
        # To-Do: Load different models!
        model = SentenceTransformer('models/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')

        logger.info("Encode & Embed entities...")

        embeddings = model.encode(list(pattern2id_1.keys()), batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True)

    elif model_type == 'supcon':
        logger.info('Load Models')
        model = ContrastivePretrainModel(len_tokenizer=len(tokenizer), proj=proj)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        logger.info("Encode & Embed entities...")

        def encode_and_embed(examples):
            with torch.no_grad():
                tokenized_input = tokenizer(examples, padding=True, truncation=True, max_length=16, return_tensors='pt')
                encoded_output = model(input_ids=tokenized_input['input_ids'],
                                       attention_mask=tokenized_input['attention_mask'])
                return encoded_output[1]

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in tqdm(range(0, len(lst), n)):
                yield lst[i:i + n]

        embeddings = []
        for examples in chunks(list(pattern2id_1.keys()), 256):
            embeddings.append(encode_and_embed(examples))
        embeddings = np.concatenate(embeddings, axis=0)


    else:
        logger.warning("Model Type {} not defined!".format(model_type))


    #embeddings = np.concatenate(embeddings)
    # Make sure that the embeddings are normalized --> cosine similarity
    logger.info('Initialize faiss index')
    d = embeddings.shape[1]
    m = 16
    nlist = int(4*math.sqrt(len(embeddings)))
    quantizer = faiss.IndexFlatIP(d)
    #faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) # 8 specifies that each sub-vector is encoded as 8 bits

    assert not faiss_index.is_trained
    logger.info('Train Faiss Index')
    no_training_records = nlist * 120 # Experiment with number of training records
    if embeddings.shape[0] < no_training_records:
        faiss_index.train(embeddings)
    else:
        train_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=no_training_records, replace=False), :]
        faiss_index.train(train_embeddings)
    assert faiss_index.is_trained

    logger.info('Add embeddings to faiss index')
    faiss_index.add(embeddings)

    logger.info("Search products...")
    faiss_index.nprobe = 10     # the number of cells (out of nlist) that are visited to perform a search --> INCREASE if possible

    D, I = faiss_index.search(embeddings, k_hits)
    logger.info('Collect search results')
    pair2sim = defaultdict(float)
    for index in tqdm(range(len(I))):
        for distance, top_id in zip(D[index], I[index]):
            if top_id > 0:
                if (1 - distance) < 0.75:
                    # Only collect pairs with high similarity
                    break

                if index == top_id:
                    continue
                elif index < top_id:
                    candidate_group_pair = (index, top_id)
                else:
                    candidate_group_pair = (top_id, index)

                pair2sim[candidate_group_pair] = (1 - distance)

    candidate_group_pairs = [k for k, _ in sorted(pair2sim.items(), key=lambda k_v: k_v[1], reverse=True)]

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

def preprocess_input(doc):
    if len(doc) == 0:
        return ''
    else:
        doc = doc[0].lower()

        stop_words = ['ebay', 'google', 'vology', 'buy', 'cheapest', 'cheap', 'core',
                      'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '&nbsp;', '& ', '', ';', '""']
        regex_list_1 = ['^dell*', '[\d\w]*\.com', '[\d\w]*\.ca', '[\d\w]*\.fr', '[\d\w]*\.de',
                        '(\d+\s*gb\s*hdd|\d+\s*gb\s*ssd)']

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
                if pattern in doc:
                    doc = doc.replace(pattern, '')
            doc = '{} {}'.format(gb_pattern[0].replace(' ', ''),
                                 doc)  # Only take the first found pattern --> might lead to problems, but we need to focus on the first 16 tokens.

        for regex in regex_list_2:
            doc = re.sub(regex, '', doc)

        doc = re.sub('\s\s+', ' ', doc)
        doc = re.sub('\s*$', '', doc)
        doc = re.sub('^\s*', '', doc)

        if len(doc) > 0:
            tokens = tokenizer.tokenize(doc)
            pattern = tokenizer.convert_tokens_to_string(tokens[:16])
        else:
            pattern = ''

        return pattern


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

    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    stop_words_x1 = ['amazon.com', 'ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']

    k_x_1 = 15
    proj_x_1 = 128
    X1_candidate_pairs = block_neural(X_1, ["title"], k_x_1, None, 'supcon',
                                      'models/supcon/X1_model_len16_trans{}.bin'.format(proj_x_1), proj_x_1)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    k_x_2 = 15
    proj_x_2 = 128
    X2_candidate_pairs = block_neural(X_2, ["name"], k_x_2, None, 'supcon',
                                      'models/supcon/X2_model_len16_trans{}.bin'.format(proj_x_2), proj_x_2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
