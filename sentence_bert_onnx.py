from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
model = AutoModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
# model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
model.eval()

# Input to the model
with torch.no_grad():
    inputs = {'input_ids': torch.ones(1, 256, dtype=torch.int64),
              'attention_mask': torch.ones(1, 256, dtype=torch.int64)}
    outputs = model(**inputs)
    # Export the model
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,  # model being run
                      (inputs['input_ids'],  # model input (or a tuple for multiple inputs)
                       inputs['attention_mask']),
                      "xtremedistil-l6-h256-uncased.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input_ids',  # the model's input names
                                   'attention_mask'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                    'attention_mask': symbolic_names})


from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions

# disable embedding layer norm optimization for better model size reduction
opt_options = BertOptimizationOptions('bert')
opt_options.enable_embed_layer_norm = False

opt_model = optimizer.optimize_model(
    'xtremedistil-l6-h256-uncased.onnx',
    'bert',
    num_heads=4,
    hidden_size=256,
    optimization_options=opt_options)
opt_model.save_model_to_file('xtremedistil-l6-h256-uncased.opt.onnx')

#import onnx
#import onnxruntime

#sess_options = onnxruntime.SessionOptions()
#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#ort_session = onnxruntime.InferenceSession("test.opt.onnx")