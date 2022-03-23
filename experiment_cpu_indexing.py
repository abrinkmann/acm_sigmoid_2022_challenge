import logging
from datetime import datetime

import click
import numpy
import pandas as pd
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)

@click.command()
@click.option('--file')
@click.option('--batch_size', type=int, default=64)
@click.option('--num_proc', type=int, default=16)
def index_entities(file, batch_size, num_proc):

    logger = logging.getLogger()
    # Load data into a pandas data frame
    df_data = pd.read_csv(file, sep=',', encoding='utf-8')
    df_data['title'] = df_data['title'].str[:64]
    df_data = df_data.groupby(by=['title'])['id'].apply(list).reset_index(name='ids')

    ds = Dataset.from_pandas(df_data)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

    def encode_and_embed(examples):
        #tokenized_output = tokenizer(examples['title'], padding="max_length", truncation=True, max_length=64)
        tokenized_output = tokenizer(examples['title'], padding=True, truncation=True, max_length=64)
        encoded_output = model(input_ids=torch.tensor(tokenized_output['input_ids']),
                               attention_mask=torch.tensor(tokenized_output['attention_mask']),
                               token_type_ids=torch.tensor(tokenized_output['token_type_ids']))
        return encoded_output['pooler_output'].detach().numpy()

    start = datetime.now()
    logger.info('Encode & embed titles!')

    # Set num proc to number of available CPUs(?)
    ds_with_embeddings = ds.map(lambda examples: {'embeddings': encode_and_embed(examples)}, batched=True,
                                batch_size=batch_size, num_proc=num_proc)

    logger.info('Add embeddings to faiss index')
    ds_with_embeddings.add_faiss_index(column='embeddings')

    end = datetime.now()
    logger.info('With HF batching: ' + str((end - start).total_seconds() / 60.0))

    first_embedding = numpy.array([ds_with_embeddings[0]['embeddings']]).astype('float32')

    entity = "Panasonic Latitude 14 B5232 - Solid Duo"

    entity_embedding = model(**tokenizer(entity, return_tensors="pt"))['pooler_output'].numpy()

    scores, samples = ds_with_embeddings.get_nearest_examples(
        "embeddings", first_embedding, k=5
    )

    print('I am here')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    index_entities()
