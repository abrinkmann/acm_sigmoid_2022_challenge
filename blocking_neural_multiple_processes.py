
import itertools
import logging
import re
import time
from collections import defaultdict
from multiprocessing import Pool, Queue, Process

import faiss
import torch

import numpy as np
from psutil import cpu_count
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def block(X, attr, expected_cand_size, k_hits, parallel, batch_sizes, configurations):  # replace with your logic.
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

    # pattern2id_1 = defaultdict(list)
    # logger.info("Group products...")
    # for i in tqdm(range(X.shape[0])):
    #     pattern2id_1[X['preprocessed'][i]].append(X['id'][i])
    #
    # goup_ids = [i for i in range(len(pattern2id_1))]
    # group2id_1 = dict(zip(goup_ids, pattern2id_1.values()))
    #
    # # Prepare pairs deduced from groups while waiting for search results
    # logger.info('Create group candidates')
    # # Add candidates from grouping
    # candidate_pairs_real_ids = []
    #
    # for ids in tqdm(group2id_1.values()):
    #     ids = list(sorted(ids))
    #     for j in range(len(ids)):
    #         for k in range(j + 1, len(ids)):
    #             candidate_pairs_real_ids.append((ids[j], ids[k]))
    #
    # candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in tqdm(range(0, len(lst), n)):
            yield lst[i:i + n]

    if parallel:
        with open('hyperparameter.txt', 'w') as f:
            f.write('Batch Size, Number of Threads, Worker, Processing time\n')
        for batch_size, configuration in itertools.product(batch_sizes, configurations):
            start = time.time()
            input_queue = Queue()
            output_queue = Queue()

            logger.info("Encode & Embed entities...")
            worker = configuration['worker']
            processes = []

            for i in range(worker):
                p = Process(target=encode_and_embed, args=(input_queue, output_queue, configuration['num_threads'], ))
                p.start()
                processes.append(p)

            for examples in chunks(list(X['preprocessed'].values), batch_size):
                input_queue.put(examples)

            pbar = tqdm(total=int(len(list(X['preprocessed'].values))/batch_size))

            embeddings = []
            while not input_queue.empty():
                while not output_queue.empty():
                    embeddings.append(output_queue.get())
                    pbar.update(1)

            input_queue.close()
            input_queue.join_thread()

            for process in processes:
                while process.is_alive():
                    while not output_queue.empty():
                        embeddings.append(output_queue.get())
                        pbar.update(1)
                process.join()

            embeddings = np.concatenate(embeddings, axis=0)

            time.sleep(0.1)
            end = time.time()
            run_time = (end - start)
            logger.info('Processing time: {} - Batch Size: {} - Num_Threads: {} - Worker: {}'.format(run_time,
                                                                                                     batch_size,
                                                                                                     configuration['num_threads'],
                                                                                                     configuration['worker']))
            with open('hyperparameter.txt', 'a') as f:
                f.write('{},{},{},{}\n'.format(batch_size, configuration['num_threads'], configuration['worker'], run_time))
            output_queue.close()
            output_queue.join_thread()

    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased", use_fast=True)
        model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

        def encode_and_embed_local(examples):
            # tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
            with torch.no_grad():
                tokenized_output = tokenizer(examples, padding=True, truncation=True, max_length=64,
                                             return_tensors='pt')
                encoded_output = model(**tokenized_output)
                result = encoded_output['pooler_output'].detach().numpy()
                return result

        embeddings = np.empty((0, 256), dtype=np.float32)
        for examples in chunks(list(X['preprocessed'].values), 256):
            embeddings = np.append(embeddings, encode_and_embed_local(examples), axis=0)


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

    #return candidate_pairs_real_ids
    return []


def encode_and_embed(input_q, output_q, num_threads):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    #torch.set_num_threads(num_threads)

    while not input_q.empty():
        examples = input_q.get()
        # tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
        with torch.no_grad():
            tokenized_output = tokenizer(examples, padding=True, truncation=True, max_length=32, return_tensors='pt')
            encoded_output = model(input_ids=tokenized_output['input_ids'],
                                   attention_mask=tokenized_output['attention_mask'],
                                   token_type_ids=tokenized_output['token_type_ids'])
            result = encoded_output['pooler_output'].detach().numpy()

            output_q.put(result)


def preprocess_input(doc):
    doc = doc[0].lower()

    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

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
    # batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    # configurations = [{'num_threads': 1, 'worker': 16}, {'num_threads': 2, 'worker': 8},
    #                   {'num_threads': 4, 'worker': 4}, {'num_threads': 8, 'worker': 2},
    #                   {'num_threads': 16, 'worker': 1}]

    batch_sizes = [128]
    configurations = [{'num_threads': 4, 'worker': 4}]

    X1_candidate_pairs = block(X_1, ["title"], expected_cand_size_X1, k_x_1, True, batch_sizes, configurations)
    # if len(X1_candidate_pairs) > expected_cand_size_X1:
    #     X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    # #X2_candidate_pairs = []
    # stop_words_x2 = []
    # k_x_2 = 2
    # X2_candidate_pairs = block(X_2, ["name"], expected_cand_size_X1, k_x_2, parallel=True)
    # if len(X2_candidate_pairs) > expected_cand_size_X2:
    #     X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]
    #
    # # save results
    # save_output(X1_candidate_pairs, X2_candidate_pairs)
    # logger = logging.getLogger()
    # logger.info('Candidates saved!')