import logging

import click
import torch
from datasets import load_dataset
import time

from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertTokenizer


@click.command()
@click.option('--file')
def index_entities(file):

    torch.set_grad_enabled(False)

    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    ds = load_dataset('csv', data_files=file, split='train')

    #start = time.time()
    #ds_with_embeddings = ds.map(lambda example: {model(**tokenizer(example["title"], return_tensors="pt",
    #                                                                              padding=True, truncation=True, max_length=64))['pooler_output']}, batched=True, batch_size=32)

    #ds_with_embeddings.add_faiss_index(column='embeddings')

    #end = time.time()
    #print('With batching: ' + str(end - start))

    ##################################

    start = time.time()

    def tokenize_function(examples):
        output = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=64)
        return output

    def encode_function(examples):
        encoded_output = model(input_ids=torch.tensor(examples['input_ids']),
                    token_type_ids=torch.tensor(examples['token_type_ids']),
                    attention_mask=torch.tensor(examples['attention_mask']))
        # Use <cls> token
        output = {'embeddings': encoded_output['pooler_output'].numpy()}
        return output

    ds_tokenized = ds.map(tokenize_function, batched=True)

    print('After tokenization')
    ds_with_embeddings = ds_tokenized.map(encode_function, batched=True, batch_size=32)
    print('After Encoding')

    ds_with_embeddings.add_faiss_index(column='embeddings')

    end = time.time()
    print('With batching: ' + str(end - start))

    ##################################

    start = time.time()
    ds_with_embeddings = ds.map(lambda example: {'embeddings': model(**tokenizer(example["title"], return_tensors="pt",
                                                                                  padding=True, truncation=True, max_length=64))['pooler_output'][:, 0].numpy()})

    ds_with_embeddings.add_faiss_index(column='embeddings')

    end = time.time()
    print('Without batching: ' + str(end - start))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    index_entities()
