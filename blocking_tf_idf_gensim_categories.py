import itertools
import logging
import re
import time
from collections import defaultdict

import gensim
from multiprocessing import Pool, Queue, Process

import numpy as np
from gensim import corpora
from psutil import cpu_count
from tqdm import tqdm
import pandas as pd
from gensim.models import TfidfModel



def block_with_bm25(X, attr, expected_cand_size, k_hits, brands, parallel):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()
    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    logger.info("Preprocessing products...")
    worker = cpu_count()
    pool = Pool(worker)
    X['preprocessed'] = pool.map(preprocess_input, tqdm(list(X[attr].values)))
    pool.close()
    pool.join()

    docbrand2pattern2id = defaultdict(lambda: defaultdict(list))
    logger.info("Group products...")
    for i in tqdm(range(X.shape[0])):
        pattern = X['preprocessed'][i]
        doc_brand = 'Null'
        for brand in brands:
            if brand in pattern:
                doc_brand = brand
                break
        docbrand2pattern2id[doc_brand][pattern].append(X['id'][i])


    # Prepare pairs deduced from groups while waiting for search results
    logger.info('Create group candidates')
    # Add candidates from grouping
    candidate_pairs_real_ids = []

    for doc_brand in tqdm(docbrand2pattern2id.keys()):
        for ids in docbrand2pattern2id[doc_brand].values():
            ids = list(sorted(ids))
            for j in range(len(ids)):
                for k in range(j + 1, len(ids)):
                    candidate_pairs_real_ids.append((ids[j], ids[k]))

    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))


    logger.info('Start search')
    if parallel:
        worker = 4
        pool = Pool(worker)

        new_candidate_pairs_real_ids = pool.starmap(search_tfidf_gensim, zip(docbrand2pattern2id.values(),
                                                                      itertools.repeat(k_hits)))

        pool.close()
        pool.join()

    else:
        # Fill input queue
        input_queue = Queue()
        output_queue = Queue()
        for doc_brand in docbrand2pattern2id.values():
            input_queue.put(doc_brand)

        worker = 3
        processes = []

        for i in range(worker):
            p = Process(target=search_tfidf_gensim, args=(input_queue, output_queue, k_hits,))
            p.start()
            processes.append(p)

        while not input_queue.empty():
            time.sleep(1)
        input_queue.close()

        while not output_queue.empty():
            new_candidate_pairs_real_ids = output_queue.get()
            candidate_pairs_real_ids.extend(new_candidate_pairs_real_ids)
            candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))

        for process in processes:
            while process.is_alive():
                while not output_queue.empty():
                    new_candidate_pairs_real_ids = output_queue.get()
                    candidate_pairs_real_ids.extend(new_candidate_pairs_real_ids)
                    candidate_pairs_real_ids = list(set(candidate_pairs_real_ids))
            process.join()

        output_queue.close()

    logger.info('Finished search')

    return candidate_pairs_real_ids


def tokenize(value):
    return value.split(' ')


def tokenize_tfidf_vectorizer(value):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tokens = token_pattern.findall(value)

    return tokens


def search_tfidf_gensim(input_queue, output_queue, k_hits):

    while not input_queue.empty():
        doc_brand = input_queue.get()
        logger = logging.getLogger()
        candidate_group_pairs = []
        tokenized_corpus = [tokenize(product) for product in doc_brand.keys()]
        dct = corpora.Dictionary(tokenized_corpus)
        abs_max_df = 30
        if len(tokenized_corpus) > abs_max_df: # fit dictionary
            dct.filter_extremes(no_below=2, no_above=abs_max_df/len(tokenized_corpus))

        corpus = [dct.doc2bow(product) for product in tokenized_corpus]

        tfidf = TfidfModel(corpus)
        #corpus_tfidf = tfidf[corpus]
        logger.info('Create Similarity Matrix')
        index = gensim.similarities.Similarity(output_prefix=None, corpus=tfidf[corpus], num_features=len(dct),
                                               num_best=k_hits)

        logger.info('Collect similarities')
        i = 0
        for sims in index:
            for hit in sims:
                top_id, _ = hit
                if i != top_id:
                    if i < top_id:
                        candidate_group_pair = (i, top_id)
                    else:
                        candidate_group_pair = (top_id, i)

                    candidate_group_pairs.append(candidate_group_pair)
            i += 1

        candidate_group_pairs = list(set(candidate_group_pairs))
        new_candidate_pairs_real_ids = []
        goup_ids = [i for i in range(len(doc_brand.values()))]
        group2id_1 = dict(zip(goup_ids, doc_brand.values()))

        for pair in candidate_group_pairs:
            real_group_ids_1 = list(sorted(group2id_1[pair[0]]))
            real_group_ids_2 = list(sorted(group2id_1[pair[1]]))

            for real_id1, real_id2 in itertools.product(real_group_ids_1, real_group_ids_2):
                if real_id1 < real_id2:
                    candidate_pair = (real_id1, real_id2)
                elif real_id1 > real_id2:
                    candidate_pair = (real_id2, real_id1)
                else:
                    continue
                new_candidate_pairs_real_ids.append(candidate_pair)

        output_queue.put(new_candidate_pairs_real_ids)
    print('Done')
    return


def preprocess_input(doc):
    doc = doc.lower()

    stop_words = ['ebay', 'google', 'vology', 'alibaba.com', 'buy', 'cheapest', 'cheap',
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ']
    regex_list = ['^dell*', '[\d\w]*\.com', '\/', '\|', '--\s', '-\s', '^-', '-$', ':\s', '\(', '\)', ',']

    doc = doc.replace('hewlett-packard', 'hp')

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
                     'miniprice.ca', 'refurbished', 'wifi', 'best', 'wholesale', 'price', 'hot', '& ', 'china']

    k_x_1 = 2
    brands_x_1 = ['vaio', 'samsung', 'fujitsu', 'lenovo', 'hp',  'asus', 'panasonic', 'toshiba',
                  'sony', 'aspire', 'dell']
    X1_candidate_pairs = block_with_bm25(X_1, "title", expected_cand_size_X1, k_x_1, brands_x_1, parallel=False)
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]

    #X2_candidate_pairs = []
    stop_words_x2 = []
    k_x_2 = 3
    brands_x_2 = ['lexar', 'kingston', 'samsung', 'sony', 'toshiba', 'sandisk', 'intenso', 'transcend']
    X2_candidate_pairs = block_with_bm25(X_2, "name", expected_cand_size_X1, k_x_2, brands_x_2, parallel=False)
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
