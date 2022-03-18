import itertools
import logging
import os

import click
import psutil
import ray
import torch
import torch.distributed as dist
from datasets import load_dataset
import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertTokenizer, training_args, pipeline

torch.set_grad_enabled(False)

@click.command()
@click.option('--file')
@click.option('--batch_size', type=int, default=16)
@click.option('--num_proc', type=int, default=2)
@click.option('--local_rank', type=int, default=0)
def index_entities(file, batch_size, num_proc, local_rank):

    logger = logging.getLogger()
    ds = load_dataset('csv', data_files=file, split='train')

    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    #setup(local_rank, 3)

    start = time.time()

    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=64)

        return tokenized_output

    @ray.remote
    def tokenize_embed(p_model, p_tokenizer, example):
        tokenized_output = p_tokenizer(example['title'], padding="max_length", truncation=True, max_length=64)
        encoded_output = p_model(input_ids=torch.tensor([tokenized_output['input_ids']]),
                               attention_mask=torch.tensor([tokenized_output['attention_mask']]),
                               token_type_ids=torch.tensor([tokenized_output['token_type_ids']]))
        return encoded_output['pooler_output'][0].detach().numpy()

    def embedding_function(examples):
        encoded_output = model(input_ids=torch.tensor(examples['input_ids']),
                               attention_mask=torch.tensor(examples['attention_mask']),
                               token_type_ids=torch.tensor(examples['token_type_ids']))
        output = {'embeddings': encoded_output['pooler_output'].detach().numpy()}
        return output

    def tokenize_and_embedding_function(examples):
        return embedding_function(tokenize_function(examples))

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in tqdm(range(0, len(lst), n)):
            yield lst['title'][i:i + n]

    #ds_with_tokens = ds.map(tokenize_function, batched=True)

    num_cpus = psutil.cpu_count(logical=True)
    print('Number of available CPUs:', num_cpus)

    # Start Ray cluster
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    model_id = ray.put(model)
    tokenizer_id = ray.put(tokenizer)

    # Add chunks --> "batches"
    embeddings = ray.get([tokenize_embed.remote(model_id, tokenizer_id, example) for example in tqdm(ds)])
    #embeddings = list(itertools.chain(*embeddings))
    ds_with_embeddings = ds.add_column('embeddings', embeddings)
        #.map(tokenize_and_embedding_function, batched=True, batch_size=batch_size)
    #logger.info(f"END PREPROCESSING on RANK {local_rank}")

    #if local_rank == 0:
    #    logger.info('Loading results from the main process')
    #    dist.barrier()
    ray.shutdown()
    ds_with_embeddings.add_faiss_index(column='embeddings')

    end = time.time()
    #cleanup()
    print('With Ray: ' + str(end - start))

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
