import logging

import click
import torch
from datasets import load_dataset
import time

from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertTokenizer


@click.command()
@click.option('--file')
@click.option('--batch_size', type=int, default=32)
@click.option('--num_proc', type=int, default=2)
def index_entities(file, batch_size, num_proc):

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

    def tokenize_and_encode_function(examples):
        tokenized_output = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=64)
        encoded_output = model(input_ids=torch.tensor(tokenized_output['input_ids']),
                               attention_mask=torch.tensor(tokenized_output['attention_mask']),
                               token_type_ids=torch.tensor(tokenized_output['token_type_ids']))
        output = {'embeddings': encoded_output['pooler_output'].detach().numpy()}
        return output

    ds_with_embeddings = ds.map(tokenize_and_encode_function, batched=True, batch_size=batch_size, num_proc=num_proc)

    print('After Tokenization & Encoding')

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
