import itertools
import logging

import click
import pandas as pd

from utils import collect_pairs_from_file


@click.command()
@click.option('--filename')
@click.option('--gt_filename')
def add_gt_cluster_to_records(filename, gt_filename):
    df_data = pd.read_csv('{}.csv'.format(filename), sep=',', encoding='utf-8')

    pairs = []
    for str_pair in collect_pairs_from_file('{}.csv'.format(gt_filename))[1:]:
        str_pair_parts = str_pair.split('-')
        pair = (int(str_pair_parts[0]), int(str_pair_parts[1]))
        pairs.append(pair)

    # Merge lists until no change happens anymore

    change = True
    print(len(pairs))
    while change:
        cluster = []
        #print(len(pairs))
        old_length = len(pairs)
        while len(pairs) > 0:
            pair = pairs.pop()
            removable_pairs = []
            for new_candidate_pair in pairs:
                matches = sum([1 for element in new_candidate_pair if element in pair])
                if matches > 0:
                    removable_pairs.append(new_candidate_pair)
                    pair = tuple(set(pair + new_candidate_pair))

            cluster.append(pair)
            pairs = [pair for pair in pairs if pair not in removable_pairs]

        pairs = cluster
        change = len(pairs) != old_length
        if change:
            print(len(pairs))
            print('new iteration')

    cluster_pair_dict = {'clu_{}'.format(i): pairs[i] for i in range(0, len(pairs))}
    id_cluster_dict = {}
    for cluster_id, pair in cluster_pair_dict.items():
        for identifier in pair:
            id_cluster_dict[identifier] = cluster_id

    df_data['cluster_{}'.format(gt_filename)] = df_data['id'].map(id_cluster_dict)
    df_data.to_csv('{}_with_cluster.csv'.format(filename), sep=',', encoding='utf-8', index=False)
    print('I am here!')






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    add_gt_cluster_to_records()