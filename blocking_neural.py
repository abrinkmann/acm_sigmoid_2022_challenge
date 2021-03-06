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
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from model_contrastive import ContrastivePretrainModel

tokenizer = AutoTokenizer.from_pretrained('tokenizer')


def load_normalization():
    """Load Normalization file"""
    normalizations = {}
    with open('normalization.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_values = line.split(',')
            normalizations[line_values[0]] = line_values[1].replace('\n', '')

    return normalizations


def block_neural(X, attrs, config, path_to_preprocessed_file, normalizations, model_path, expected_candidate_size):
    """
    This function performs blocking using neural retrieval
    :param X: dataframe
    :param attrs: attributes used for blocking
    :param config: dictionary containing the hyper parameters of the blocking system
    :param path_to_preprocessed_file: intermediate output of preprocessed data, which is used to debug the preprocessing
    :param normalizations: dictionary containing the normalizations applied during preprocessing
    :param model_path: path to model used for embeddings
    :param expected_candidate_size: expected number of candidate tuples
    :return: candidate set of tuple pairs
    """

    logger = logging.getLogger()

    logger.info("Pre-process products...")

    worker = cpu_count()
    pool = Pool(worker)
    # Pre-process tuples
    X['preprocessed'] = pool.starmap(preprocess_input, zip(list(X[attrs].values), repeat(normalizations),
                                                           repeat(config['seq_length'])))

    if path_to_preprocessed_file is not None:
        # Save pre-processed data for debugging purposes
        X['tokens'] = pool.map(tokenize_input, tqdm(list(X['preprocessed'].values)))
        X.to_csv(path_to_preprocessed_file, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_MINIMAL)

    pool.close()
    pool.join()

    logger.info("Group products...")
    # Group products by preprocessed surface form
    pattern2id_1 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        pattern2id_1[X['preprocessed'][i]].append(X['id'][i])

    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates...')

    group_ids = [i for i in range(len(pattern2id_1))]
    group2id_1 = dict(zip(group_ids, pattern2id_1.values()))

    # Create first candidates in subprocess while models are loaded
    candidate_pairs_real_ids = []
    for ids in tqdm(group2id_1.values()):
        ids = list(sorted(ids))
        for j in range(len(ids)):
            for k in range(j + 1, len(ids)):
                candidate_pairs_real_ids.append((ids[j], ids[k]))

    logger.info('Load Models...')

    model = ContrastivePretrainModel(len_tokenizer=len(tokenizer), proj=config['proj'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    logger.info("Encode & Embed products...")

    def encode_and_embed(docs):
        with torch.no_grad():
            tokenized_input = tokenizer(docs, padding=True, truncation=True, max_length=config['seq_length'],
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
        #  Create embeddings in batches
        embeddings.append(encode_and_embed(examples))
    embeddings = np.concatenate(embeddings, axis=0)

    logger.info('Initialize FAISS index...')
    # Use FAISS for efficient similarity search and clustering of dense vectors
    # Visit https://github.com/facebookresearch/faiss/wiki/ for further details
    d = embeddings.shape[1]  # embedding dimensions used for semantic search
    m = 16  # number of sub-quantizers
    nlist = int(config['nlist_factor'] * math.sqrt(embeddings.shape[0]))  # Number created Voronoi cells
    quantizer = faiss.IndexFlatIP(d)  # base quantizer
    # faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    faiss_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 specifies that each sub-vector is encoded as 8 bits

    assert not faiss_index.is_trained
    logger.info('Train FAISS Index...')
    no_training_records = nlist * config['train_data_factor']
    if embeddings.shape[0] < no_training_records:
        faiss_index.train(embeddings)
    else:
        # Select a random subset of embeddings for training faiss
        train_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=no_training_records, replace=False), :]
        faiss_index.train(train_embeddings)
    assert faiss_index.is_trained

    logger.info('Add embeddings to FAISS index...')
    faiss_index.add(embeddings)

    logger.info("Search products...")
    faiss_index.nprobe = config['nprobe']  # Number visited Voronoi cells

    # Search for top k nearest neighbours per tuple
    D, I = faiss_index.search(embeddings, config['k'])
    logger.info('Collect search results...')
    # Collect only group candidates with a similarity above a similarity threshold of 0.4 (Combine top-k & range-search)
    pair2sim = defaultdict(float)
    for group_index in tqdm(range(len(I))):
        for distance, top_group_id in zip(D[group_index], I[group_index]):
            if top_group_id > -1:
                if (1 - distance) < 0.4:
                    break
                if group_index == top_group_id:
                    continue
                elif group_index < top_group_id:
                    candidate_group_pair = (group_index, top_group_id)
                else:
                    candidate_group_pair = (top_group_id, group_index)

                # Collect max found similarity for candidate pair
                pair2sim[candidate_group_pair] = max(1 - distance, pair2sim[candidate_group_pair])

    if config['transitive_closure']:
        logger.info('Determine transitive pairs...')
        pair2sim = determine_transitive_candidates(pair2sim)

    if config['jaccard_reranking']:
        logger.info('Perform Jaccard re-ranking...')

        # Tokenize Group surface forms by white-space
        tokenized_patterns = [split_by_whitespace(pattern) for pattern in pattern2id_1.keys()]

        for pair in pair2sim.keys():
            # Calculate Jaccard similarity for all candidate pairs
            tokens_1 = tokenized_patterns[pair[0]]
            tokens_2 = tokenized_patterns[pair[1]]
            jaccard_sim = len(tokens_1.intersection(tokens_2)) / max(len(tokens_1), len(tokens_2))

            # Update pair similarity with jaccard similarity
            pair2sim[pair] = 0.5 * pair2sim[pair] + 0.5 * jaccard_sim

    # Sort candidate group pairs by similarity
    candidate_group_pairs = [k for k, _ in sorted(pair2sim.items(), key=lambda k_v: k_v[1], reverse=True)]

    logger.info('Map GroupIds to real ids...')
    for pair in tqdm(candidate_group_pairs):
        if len(candidate_pairs_real_ids) > expected_candidate_size:
            # Stop creating new candidate pairs if the expected number of
            break

        # Determine real ids
        real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
        real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

        for real_id1, real_id2 in product(real_group_ids_1, real_group_ids_2):
            # Create pairs as expected by the organizers
            if real_id1 < real_id2:
                candidate_pair = (real_id1, real_id2)
            elif real_id1 > real_id2:
                candidate_pair = (real_id2, real_id1)
            else:
                continue
            candidate_pairs_real_ids.append(candidate_pair)

    return candidate_pairs_real_ids


def preprocess_input(docs, normalizations, seq_length):
    """Pre-process input docs and make sure that as little precision as possible is lost by this operation
        :param docs: input doc values of the original data set
        :param normalizations: dictionary containing the normalizations applied during preprocessing
        :param seq_length: sequence length used by tokenizer
        :return pre-processed input doc
    """
    if len(docs) == 0:
        return ''
    else:
        # Join multiple attribute values if multiple attributes are used by blocking system
        doc = ' '.join([str(value) for value in docs if
                        type(value) is str or (type(value) is float and not np.isnan(value))]).lower()

        # Manually selected stop words
        stop_words = ['ebay', 'google', 'vology', 'buy', 'cheapest', 'foto de angelis', 'cheap', 'core',
                      'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '\'\'', '"', '\\\\n',
                      'tesco direct', 'color', ' y ', ' et ', 'tipo a', 'type-a', 'type a', 'inform??tica', ' de ',
                      ' con ', 'newest', ' new', ' ram ', '64-bit', '32-bit', 'accessories', 'series', 'touchscreen',
                      'product', 'customized']

        stop_signs = ['&nbsp;', '&quot;', '&amp;', ',', ';', '-', ':', '|', '/', '(', ')', '/', '&']

        regex_list_1 = ['^dell*', '[\d\w]*\.com', '[\d\w]*\.ca', '[\d\w]*\.fr', '[\d\w]*\.de', '[\d\w]*\.es',
                        '(\d+\s*gb\s*hdd|\d+\s*gb\s*ssd)', '\\\\n']

        # Remove stop words
        for stop_word in stop_words:
            doc = doc.replace(stop_word, ' ')

        # Remove stop signs
        for stop_sign in stop_signs:
            doc = doc.replace(stop_sign, ' ')

        # Remove special regexes
        for regex in regex_list_1:
            doc = re.sub(regex, '', doc)

        # Move GB pattern to beginning of doc
        # Through manual inspection we found that these gb patterns are important for the disambiguation of products.
        # So we try to make sure that they appear in the sequence even after truncation.
        gb_pattern = re.findall('(d+\s*gbbeuk|\d+\s*gbbeu|\d+\s*gb|\d+\s*go|\d+\s*bbeu|\d+\s*gabeu)', doc)

        if len(gb_pattern) > 0:
            # Sort patterns and move only the first pattern if it does not start with a 0 - example: 0GB
            gb_pattern.sort()
            while len(gb_pattern) > 0 and gb_pattern[0][0] == '0':
                gb_pattern.remove(gb_pattern[0])

            if len(gb_pattern) > 0:
                doc = re.sub('(d+\s*gbbeuk|\d+\s*gbbeu|\d+\s*gb|\d+\s*go|\d+\s*bbeu|\d+\s*gabeu)', ' ', doc)
                doc = '{} {}'.format(
                    gb_pattern[0].replace(' ', '').replace('go', 'gb').replace('gbbeuk', 'gb').replace('gbbeu',
                                                                                                       'gb').replace(
                        'bbeu', 'gb'),
                    doc)  # Only take the first found pattern

        doc = re.sub('\s\s+', ' ', doc)

        if normalizations is not None:
            # Apply normalizations as specified in the normalization dictionary
            # The normalizations are selected through manual data profiling
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
            # Tokenize normalized doc and truncate at max sequence length.
            tokens = tokenizer.tokenize(doc)
            pattern = tokenizer.convert_tokens_to_string(tokens[:seq_length])
        else:
            pattern = ''

        return pattern


def determine_transitive_candidates(pairs2sim):
    """Determine transitive matches based on found candidates
        :param pairs2sim: Found pairs
        :return Updated dictionary with pairs
    """
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


def split_by_whitespace(doc):
    """Split doc by whitespace and return set of tokens
    :param doc: string value which will be tokenized
    :return token set
    """
    return set(doc.split(' '))


def tokenize_input(doc):
    """Tokenize doc using the loaded tokenizer
    :param doc: string value which will be tokenized
    """
    tokens = tokenizer.tokenize(doc)
    pattern = tokenizer.convert_tokens_to_string(tokens[:32])
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
    logger = logging.getLogger()

    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # Local Testing - COMMENT FOR SUBMISSION!
    # logger.warning('NOT A REAL SUBMISSION!')
    # expected_cand_size_X1 = 2814
    # expected_cand_size_X2 = 4392

    X_1 = pd.read_csv("X1.csv")
    X_2 = pd.read_csv("X2.csv")

    # Specify blocking system configuration for data set X1
    configuration_x_1 = {'seq_length': 28, 'proj': 32, 'k': 30,
                         'nlist_factor': 4, 'train_data_factor': 200, 'nprobe': 20,
                         'transitive_closure': False, 'jaccard_reranking': True}
    normalizations_x_1 = load_normalization()
    X1_candidate_pairs = block_neural(X_1, ["title"], configuration_x_1, None, normalizations_x_1,
                                      'models/supcon/len{}/X1_model_len{}_trans{}_with_computers_lower_lr.bin'.format(
                                          configuration_x_1['seq_length'], configuration_x_1['seq_length'],
                                          configuration_x_1['proj']), expected_cand_size_X1)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    # Specify blocking system configuration for data set X2
    configuration_x_2 = {'seq_length': 24, 'proj': 32, 'k': 30,
                         'nlist_factor': 4, 'train_data_factor': 200, 'nprobe': 20,
                         'transitive_closure': False, 'jaccard_reranking': True}
    normalizations_x_2 = normalizations_x_1
    X2_candidate_pairs = block_neural(X_2, ["name"], configuration_x_2, None, normalizations_x_2,
                                      'models/supcon/len{}/X2_model_len{}_trans{}_with_computers.bin'.format(
                                          configuration_x_2['seq_length'], configuration_x_2['seq_length'],
                                          configuration_x_2['proj']), expected_cand_size_X2)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)

    if expected_cand_size_X1 != 1000000 or expected_cand_size_X2 != 2000000:
        logger.warning('NOT A REAL SUBMISSION!')
    logger.info('Candidates saved!')
