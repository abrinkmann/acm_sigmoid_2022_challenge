from transformers import AutoTokenizer, AutoModel
import torch
from txtai.pipeline import HFOnnx

onnx = HFOnnx()

#tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
#model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
# model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
#model.eval()

embeddings = onnx("ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32", "pooling", "xtremedistil-l6-h256-uncased.onnx", quantize=False)

from onnxruntime.transformers import optimizer

optimized_model = optimizer.optimize_model('xtremedistil-l6-h256-uncased.onnx', model_type='bert', num_heads=8, hidden_size=256)
optimized_model.save_model_to_file('xtremedistil-l6-h256-uncased.opt.onnx')


#import onnx
#import onnxruntime

#sess_options = onnxruntime.SessionOptions()
#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#ort_session = onnxruntime.InferenceSession("test.opt.onnx")