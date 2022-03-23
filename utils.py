from tqdm import tqdm


def collect_pairs_from_file(file_name):
    pairs = []
    with open(file_name) as file:
        for line in tqdm(file.readlines()):
            pair_values = line.replace('\n', '').split(',')
            pairs.append('{}-{}'.format(pair_values[0], pair_values[1]))
    return pairs