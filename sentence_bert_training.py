import logging
import math

from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
import click
from transformers import AutoConfig, AutoTokenizer

from src.models.xtremdistil_sem_search.modelling_xtremdistil_sem_search import XtremDistilSemSearch


@click.command()
@click.option('--model_name', default='microsoft/xtremedistil-l6-h256-uncased')
@click.option('--pooling', default='mean')
@click.option('--loss', default='cosine')
@click.option('--num_epochs', type=int, default=1)
def sbert_finetuning(model_name, pooling, loss, num_epochs):
    logger = logging.getLogger()
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #To-Do: Properly introduce configuration
    config.projection_size = 32
    model = XtremDistilSemSearch(config)

    tokens = ['lenovo','thinkpad','elitebook', 'toshiba', 'asus', 'acer', 'lexar', 'sandisk', 'tesco', 'intenso', 'transcend']
    tokenizer.add_tokens(tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer.tokenizer))

    #
    # model = SentenceTransformer(modules=[model])
    #
    # # Convert the dataset to a DataLoader ready for training
    # logger.info("Read dataset")
    #
    # train_samples = []
    # dev_samples = []
    # # test_samples = []
    # dataset_path = '../training_sbert.csv'
    # df_sbert = pd.read_csv(dataset_path, sep=',', encoding='utf-8')
    #
    # for index, row in df_sbert.iterrows():
    #     score = float(row['score'])
    #     inp_example = InputExample(texts=[row['entity1'], row['entity2']], label=score)
    #
    #     if row['split'] == 'dev':
    #         dev_samples.append(inp_example)
    #     # elif row['split'] == 'test':
    #     #    test_samples.append(inp_example)
    #     else:
    #         train_samples.append(inp_example)
    #
    # train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
    #
    # train_loss = losses.CosineSimilarityLoss(model=model)
    #
    # logging.info("Read Training dev dataset")
    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
    #
    # # Configure the training. We skip evaluation in this example
    # warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    # logging.info("Warmup-steps: {}".format(warmup_steps))
    #
    # # Train the model
    # model_save_path = 'sbert_xtremedistil-l6-h256-uncased_{}_{}'.format(pooling, loss)
    # model.fit(train_objectives=[(train_dataloader, train_loss)],
    #           evaluator=evaluator,
    #           epochs=num_epochs,
    #           evaluation_steps=1000,
    #           warmup_steps=warmup_steps,
    #           output_path=model_save_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    sbert_finetuning()
