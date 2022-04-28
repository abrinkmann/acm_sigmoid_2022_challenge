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
from model_contrastive import ContrastivePretrainModel

tokenizer = AutoTokenizer.from_pretrained('models/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')

def load_normalization():
    """Load Normalization file - Especially for D2"""
    normalizations = {}
    with open('normalization.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_values = line.split(',')
            normalizations[line_values[0]] = line_values[1].replace('\n','')

    return normalizations

#normalizations = {}


def block_neural(X, attr, k_hits, path_to_preprocessed_file, norm, model_type, model_path, seq_length, proj, transitive_closure):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()

    logger.info("Preprocessing products...")

    worker = cpu_count()
    pool = Pool(worker)
    X['preprocessed'] = pool.starmap(preprocess_input, zip(list(X[attr].values), repeat(norm), repeat(seq_length)))

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

    if model_type == 'sbert':
        logger.info('Load Models')
        # To-Do: Load different models! --> Add citation for sentence bert
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
                tokenized_input = tokenizer(examples, padding=True, truncation=True, max_length=seq_length,
                                            return_tensors='pt')
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

    # embeddings = np.concatenate(embeddings)
    # Make sure that the embeddings are normalized --> cosine similarity
    logger.info('Initialize faiss index')
    d = embeddings.shape[1]
    m = 16
    nlist = int(4 * math.sqrt(embeddings.shape[0]))
    quantizer = faiss.IndexFlatIP(d)
    #faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 specifies that each sub-vector is encoded as 8 bits

    assert not faiss_index.is_trained
    logger.info('Train Faiss Index')
    no_training_records = nlist * 40  # Experiment with number of training records
    if embeddings.shape[0] < no_training_records:
        faiss_index.train(embeddings)
    else:
        train_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=no_training_records, replace=False), :]
        faiss_index.train(train_embeddings)
    assert faiss_index.is_trained

    logger.info('Add embeddings to faiss index')
    faiss_index.add(embeddings)

    logger.info("Search products...")
    faiss_index.nprobe = 10  # the number of cells (out of nlist) that are visited to perform a search --> INCREASE if possible

    D, I = faiss_index.search(embeddings, k_hits)
    logger.info('Collect search results')
    pair2sim = defaultdict(float)
    for index in tqdm(range(len(I))):
        for distance, top_id in zip(D[index], I[index]):
            if top_id > -1:
                if (1 - distance) < 0.1:
                    break
                if index == top_id:
                    continue
                elif index < top_id:
                    candidate_group_pair = (index, top_id)
                else:
                    candidate_group_pair = (top_id, index)

                pair2sim[candidate_group_pair] = max((1 - distance), pair2sim[candidate_group_pair])

    if transitive_closure:
        logger.info('Determine transitive pairs')
        pair2sim = determine_transitive_matches(pair2sim)

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


def preprocess_input(docs, normalizations, seq_length):
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

        if len(doc) > 0:
            tokens = tokenizer.tokenize(doc)
            pattern = tokenizer.convert_tokens_to_string(tokens[:seq_length])
        else:
            pattern = ''

        return pattern


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
    # logger.warning('NOT A REAL SUBMISSION!')
    # expected_cand_size_X1 = 2814
    # expected_cand_size_X2 = 4392

    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    k_x_1 = 30
    seq_length_x_1 = 32
    proj_x_1 = 16
    normalizations_x_1 = load_normalization()
    #cluster_size_threshold_x1 = None
    transitive_closure_x_1 = False
    X1_candidate_pairs = block_neural(X_1, ["title"], k_x_1, None, normalizations_x_1, 'supcon',
                                      'models/supcon/len{}/X1_model_len{}_trans{}.bin'.format(seq_length_x_1, seq_length_x_1,
                                                                                              proj_x_1), seq_length_x_1, proj_x_1, transitive_closure_x_1)
    #X1_candidate_pairs = block_with_attr(X_1, "title")
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    k_x_2 = 30
    seq_length_x_2 = 24
    proj_x_2 = 16
    normalizations_x_2 = normalizations_x_1
    #cluster_size_threshold_x2 = None
    transitive_closure_x_2 = False
    X2_candidate_pairs = block_neural(X_2, ["name"], k_x_2, None, normalizations_x_2, 'supcon',
                                      'models/supcon/len{}/X2_model_len{}_trans{}.bin'.format(seq_length_x_2, seq_length_x_2,
                                                                                              proj_x_2), seq_length_x_2, proj_x_2, transitive_closure_x_2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)

    if expected_cand_size_X1 < 1000000:
        logger.warning('NOT A REAL SUBMISSION!')
    logger.info('Candidates saved!')
