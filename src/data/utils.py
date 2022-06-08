import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

from copy import deepcopy

import re
PATTERN1 = re.compile("\"@\S+\s+")
PATTERN2 = re.compile("\s+")

def clean_string_wdcv2(words):
    if not words:
        return None
    words = words.partition('"')[2]
    words = words.rpartition('"')[0]
    words = re.sub(PATTERN1, ' ', words)
    words = re.sub(PATTERN2, ' ', words)
    words = words.replace('"', '')
    words = words.strip()
    return words

def clean_specTableContent_wdcv2(words):
    if not words:
        return None
    words = re.sub(PATTERN2, ' ', words)
    words = words.strip()
    return words