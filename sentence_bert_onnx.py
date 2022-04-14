import optimum as optimum
import torch
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoTokenizer, AutoModel

# model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
optimization_config = OptimizationConfig(optimization_level=99)

model = optimum.onnxruntime.ORTOptimizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased", feature='default')

model.export('xtremedistil-l6-h256-uncased.onnx', 'xtremedistil-l6-h256-uncased.opt.onnx', optimization_config)