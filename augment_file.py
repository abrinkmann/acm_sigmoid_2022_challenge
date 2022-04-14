
# Create an augmented file with roughly 1.000.000 lines
from random import randint

import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm


X = pd.read_csv('../X1.csv')
extended_X = X.copy()
print(X.columns)
highest_id = max(X['id'])
pbar = tqdm(total=1000000)
new_rows = []
while len(new_rows) < 1000000:
    random_int = randint(0, X.shape[0] - 1)
    augmenters = [naw.RandomWordAug(action="swap", aug_p=0.1), naw.RandomWordAug(aug_p=0.1),
                  naw.RandomWordAug(action="crop", aug_p=0.1), naw.RandomWordAug(action="substitute", aug_p=0.1)]

    for augmenter in augmenters:
        random_row = X.iloc[random_int].copy()
        if type(random_row['title']) is str:
            new_row = {}
            try:
                new_row['title'] = augmenter.augment(str(random_row['title']))
                new_row['id'] = highest_id

                highest_id += 1

                new_rows.append(new_row)
                pbar.update(1)
            except ValueError:
                print('Value Error!')

if len(new_rows) > 0:
    new_X = pd.DataFrame(new_rows)
    extended_X = pd.concat([extended_X, new_X], axis=0)

pbar.close()
extended_X.to_csv('X1_extended.csv', index=False)
