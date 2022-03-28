import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm
import pandas as pd



def block_with_elastic(X, index_name, attr):  # replace with your logic.
    '''
    This function performs blocking using elastic search
    :param X: dataframe
    :return: candidate set of tuple pairs
    '''

    logger = logging.getLogger()
    # connect to elastic search - Use existing elastic search instance for now
    _es = Elasticsearch(['http://{}:9200'.format(os.environ['ES_CLIENT'])])
    while not _es.ping():
        # Wait until es is up and running
        time.sleep(5)

    logger.info('Elastic Search is up and running!')
    # Delete index - ONLY TESTING PURPOSES!
    _es.indices.delete(index=index_name, ignore=[404])

    logger.info("Indexing products...")
    number_of_products = len(X)
    progress = tqdm(unit="indexed products", total=len(X))
    successes = 0
    # for index, row in X.iterrows():
    #     doc = {key: value for key, value in row.to_dict().items() if not (type(value) is float and np.isnan(value))}
    #     _es.index(index=index_name, id=index, document=doc)

    for ok, action in streaming_bulk(client=_es, index=index_name, actions=generate_products(X),):
        progress.update(1)
        successes += ok
    logger.info("Indexed %d/%d documents" % (successes, number_of_products))
    progress.close()

    _es.cluster.health(wait_for_status='green', request_timeout=30)
    # Search matches for each candidate
    logger.info("Start querying ES")
    candidate_pairs = set()

    pool = Pool(10)
    results = []

    progress_2 = tqdm(unit="searched products", total=len(X))
    for index, row in X.iterrows():
        results.append(
            pool.apply_async(query_elastic, (index_name, index, row.to_dict(), attr,)))

    successes = 0
    while len(results) > 0:
        collected_results = []
        for result in results:
            if result.ready():
                candidate_pairs.update(result.get())
                collected_results.append(result)
                progress_2.update(1)
                successes += 1

        # Remove collected results from list of results
        results = [result for result in results if result not in collected_results]

    logger.info("Searched %d/%d documents" % (successes, number_of_products))
    progress_2.close()
    pool.close()
    pool.join()

    candidate_pairs = determine_transitive_matches(list(candidate_pairs))

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # compute jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_sim = len(s1.intersection(s2)) / max(len(s1), len(s2))
        if jaccard_sim > 0.15:
            jaccard_similarities.append(jaccard_sim)
            # get real ids
            real_id1 = X['id'][id1]
            real_id2 = X['id'][id2]
            if real_id1 < real_id2:  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
                candidate_pairs_real_ids.append((real_id1, real_id2))
            else:
                candidate_pairs_real_ids.append((real_id2, real_id1))

    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids
    #return candidate_pairs_real_ids

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

def generate_products(X):
    for index, row in X.iterrows():
        doc = { key:value for key, value in row.to_dict().items() if not (type(value) is float and np.isnan(value))}
        action = {'_id': index, '_source': doc}
        yield action


def query_elastic(index_name, index, search_dict, attr_name):
    """Query elastic search"""
    _es = Elasticsearch(['http://{}:9200'.format(os.environ['ES_CLIENT'])])
    should_match_list = [{"match": {attr.lower(): search_dict[attr]}} for attr in search_dict
                         if attr != 'id' and not (type(search_dict[attr]) is float and np.isnan(search_dict[attr]))]
    query_body = {
        "bool": {"should": should_match_list}
    }
    search_result = _es.search(query=query_body, index=index_name, request_timeout=30)
    max_score = search_result.body['hits']['max_score']
    candidate_pairs = []
    for hit in search_result.body['hits']['hits']:
        if hit['_score'] / max_score < 0.33:
            # Only consider hits with a normalized similarity above 0.33
            break

        s1 = set(search_dict[attr_name].lower().split())
        s2 = set(hit['_source'][attr_name].lower().split())
        jaccard_sim = len(s1.intersection(s2)) / max(len(s1), len(s2))
        if jaccard_sim <= 0.2:
            # Filter unlikely pairs with jaccard below or equal to 0.2
            break

        result_id = int(hit['_id'])

        # Order ids by size --> Do not add rows with same id.
        if index < result_id:
            candidate_pairs.append((index, result_id))
        elif index > result_id:
            candidate_pairs.append((result_id, index))

    return candidate_pairs


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
    # read the datasets
    X1 = pd.read_csv("X1.csv")
    X2 = pd.read_csv("X2.csv")

    # perform blocking
    X1_candidate_pairs = block_with_elastic(X1, 'x1', attr="title")
    X2_candidate_pairs = block_with_elastic(X2, 'x2', attr="name")

    # save results
    save_output(X1_candidate_pairs, X2_candidate_pairs)
    logger = logging.getLogger()
    logger.info('Candidates saved!')
