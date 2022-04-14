from onnxruntime import InferenceSession
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from txtai.embeddings import Embeddings
import torch

from txtai.pipeline import HFOnnx
onnx = HFOnnx()

print('Load Model')
embeddings = onnx("ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32", "pooling", "embeddings.onnx", quantize=True)
sentence_model = SentenceTransformer('ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')
tokenizer = AutoTokenizer.from_pretrained('ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')
#classification = AutoModel.from_pretrained('sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32/2_Dense')
from sklearn.metrics.pairwise import cosine_similarity

print('Start onnx session')
session = InferenceSession("embeddings.onnx", providers=['CPUExecutionProvider'])

tokens = tokenizer(["I am happy", "I am glad"], return_tensors="np")

outputs = session.run(None, dict(tokens))[0]

print(outputs)
print(cosine_similarity(outputs))

