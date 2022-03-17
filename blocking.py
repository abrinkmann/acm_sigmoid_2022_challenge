
import torch
from datasets import load_dataset
import time

from transformers import AutoTokenizer, AutoModel, BertTokenizerFast




torch.set_grad_enabled(False)

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

ds = load_dataset('csv', data_files='X1_extended.csv', split='train')

#start = time.time()
#ds_with_embeddings = ds.map(lambda example: {model(**tokenizer(example["title"], return_tensors="pt",
#                                                                              padding=True, truncation=True, max_length=64))['pooler_output']}, batched=True, batch_size=32)

#ds_with_embeddings.add_faiss_index(column='embeddings')

#end = time.time()
#print('With batching: ' + str(end - start))

##################################

start = time.time()
ds_with_embeddings = ds.map(lambda example: {'embeddings': model(**tokenizer(example["title"], return_tensors="pt",
                                                                              padding=True, truncation=True, max_length=64))['pooler_output'][:, 0].numpy()})

ds_with_embeddings.add_faiss_index(column='embeddings')

end = time.time()
print('Without batching: ' + str(end - start))

