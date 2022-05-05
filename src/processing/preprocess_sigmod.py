import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from pathlib import Path
import shutil
import os

from copy import deepcopy

import re

from transformers import AutoTokenizer, AutoConfig

from pdb import set_trace

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[str(identifier)]
    except KeyError:
        result = cluster_id_amount
    return result

def load_normalization():
    """Load Normalization file - Especially for D2"""
    normalizations = {}
    with open('../../normalization.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_values = line.split(',')
            normalizations[line_values[0]] = line_values[1].replace('\n','')

    return normalizations
    
def build_buckets(pairs):
    bucket_list = []
    for i, row in pairs.iterrows():
        left = f'{row["lid"]}'
        right = f'{row["rid"]}'
        found = False
        for bucket in bucket_list:
            if left in bucket:
                bucket.add(right)
                found = True
                break
            elif right in bucket:
                bucket.add(left)
                found = True
                break
        if not found:
            bucket_list.append(set([left, right]))

    merging = True
    while merging:
        merging=False
        for i,group in enumerate(bucket_list):
            merged = next((g for g in bucket_list[i+1:] if g.intersection(group)),None)
            if not merged: continue
            group.update(merged)
            bucket_list.remove(merged)
            merging = True
            
    return bucket_list

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

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/xtremedistil-l6-h256-uncased', additional_special_tokens=('lenovo','thinkpad','elitebook', 'toshiba', 'asus', 'acer', 'lexar', 'sandisk', 'tesco', 'intenso', 'transcend'))
    
    # process dataset X1
    data_1 = pd.read_csv('../../X1.csv')
    labels_1 = pd.read_csv('../../Y1.csv')

    bucket_list = build_buckets(labels_1)

    cluster_id_dict = {}
    cluster_id_amount = len(bucket_list)

    for i, id_set in enumerate(bucket_list):
        for v in id_set:
            cluster_id_dict[v] = i
            
    data_1['cluster_id'] = data_1['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))

    normalizations_x1 = load_normalization()

    seq_length = 28
    data_1['features'] = data_1[['title']].apply(preprocess_input, normalizations=normalizations_x1, seq_length=seq_length, axis=1)

    single_entities = data_1[data_1['cluster_id'] == cluster_id_amount].copy()
    single_entities = single_entities.reset_index(drop=True)
    single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index

    data_1 = data_1.set_index('id', drop=False)

    data_1 = data_1.drop(single_entities['id'])
    data_1 = data_1.append(single_entities)
    data_1 = data_1.reset_index(drop=True)

    data_1 = data_1[['id', 'features', 'cluster_id']]

    os.makedirs(os.path.dirname(f'../../data/processed/blocking-sigmod-1/'), exist_ok=True)
    data_1.to_pickle(f'../../data/processed/blocking-sigmod-1/blocking-sigmod-1-train.pkl.gz', compression='gzip')
       
    # process dataset X2
    data_2 = pd.read_csv('../../X2.csv')
    labels_2 = pd.read_csv('../../Y2.csv')
    data_2.head()
    
    bucket_list = build_buckets(labels_2)

    cluster_id_dict = {}
    cluster_id_amount = len(bucket_list)

    for i, id_set in enumerate(bucket_list):
        for v in id_set:
            cluster_id_dict[v] = i
            
    data_2['cluster_id'] = data_2['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))

    normalizations_x2 = load_normalization()

    seq_length = 24
    data_2['features'] = data_2[['name']].apply(preprocess_input, normalizations=normalizations_x2, seq_length=seq_length, axis=1)

    single_entities = data_2[data_2['cluster_id'] == cluster_id_amount].copy()
    single_entities = single_entities.reset_index(drop=True)

    single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index

    data_2 = data_2.set_index('id', drop=False)

    data_2 = data_2.drop(single_entities['id'])
    data_2 = data_2.append(single_entities)
    data_2 = data_2.reset_index(drop=True)

    data_2 = data_2[['id', 'features', 'cluster_id']]

    os.makedirs(os.path.dirname(f'../../data/processed/blocking-sigmod-2/'), exist_ok=True)
    data_2.to_pickle(f'../../data/processed/blocking-sigmod-2/blocking-sigmod-2-train.pkl.gz', compression='gzip')
    
    # Process additional data for contrastive learning
    data = pd.read_pickle('../../data/interim/wdc-lspc/preprocessed_english_corpus.pkl.gz')
    
    relevant_cols = ['id', 'features', 'cluster_id']
    categories = ['computers_only_new_15']

    out_path = f'../../data/processed/wdc-lspc/'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    seq_lengths = [24, 28]

    for seq_length in seq_lengths:
        for category in categories:

            ids = pd.read_json(f'../../data/raw/wdc-lspc/pre-training_{category}.json.gz', lines=True)

            relevant_ids = set()
            relevant_ids.update(ids['id_left'])
            relevant_ids.update(ids['id_right'])

            data_selection = data[data['id'].isin(relevant_ids)].copy()

            normalizations = load_normalization()

            data_selection['features'] = data_selection[['brand', 'title']].apply(preprocess_input, normalizations=normalizations, seq_length=seq_length, axis=1)
            data_selection = data_selection[relevant_cols]
            data_selection = data_selection.reset_index(drop=True)
            data_selection.to_pickle(f'{out_path}{category}_train_sigmod_{seq_length}.pkl.gz')
        
    data_selection.head()