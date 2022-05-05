import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import os

from src.data import utils

PREPROCESS = True
DEDUP = True

if __name__ == '__main__':

    print('PREPROCESSING CORPUS')

    corpus = pd.read_json('../../data/raw/wdc-lspc/offers_corpus_english_v2_non_norm.json.gz', lines=True)

    # preprocess english corpus

    if PREPROCESS:
        print('BUILDING PREPROCESSED CORPUS...')
        corpus['title'] = corpus['title'].apply(utils.clean_string_wdcv2)
        corpus['description'] = corpus['description'].apply(utils.clean_string_wdcv2)
        corpus['brand'] = corpus['brand'].apply(utils.clean_string_wdcv2)
        corpus['price'] = corpus['price'].apply(utils.clean_string_wdcv2)
        corpus['specTableContent'] = corpus['specTableContent'].apply(utils.clean_specTableContent_wdcv2)

        os.makedirs(os.path.dirname('../../data/interim/wdc-lspc/'), exist_ok=True)
        corpus.to_pickle('../../data/interim/wdc-lspc/preprocessed_english_corpus.pkl.gz')
        print('FINISHED BUILDING PREPROCESSED CORPUS...')
