# Load Data
from tqdm import tqdm

from utils import collect_pairs_from_file

print('Load pairs')
output_pairs = collect_pairs_from_file('../output.csv')
y1_pairs = collect_pairs_from_file('../Y1.csv')
y2_pairs = collect_pairs_from_file('../Y2.csv')

# Remove 0,0 pairs from output_pairs
output_pairs = [output_pair for output_pair in output_pairs if output_pair != '0-0']

print('Search for pairs')
found_y1_pairs = [pair for pair in tqdm(y1_pairs) if pair in output_pairs]
found_y2_pairs = [pair for pair in tqdm(y2_pairs) if pair in output_pairs]

recall_y1 = len(found_y1_pairs)/ len(y1_pairs)
recall_y2 = len(found_y2_pairs)/ len(y2_pairs)
recall = (len(found_y1_pairs) + len(found_y2_pairs)) / (len(y1_pairs) + len(y2_pairs))

precision = (len(found_y1_pairs) + len(found_y2_pairs)) / len(output_pairs)

print('Recall: {}'.format(round(recall, 3)))
print('Recall Y1: {}'.format(round(recall_y1, 3)))
print('Recall Y2: {}'.format(round(recall_y2, 3)))

print('Precision: {}'.format(round(precision, 3)))
