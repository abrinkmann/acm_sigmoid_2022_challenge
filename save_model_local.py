from transformers import AutoModel

model = AutoModel.from_pretrained('microsoft/xtremedistil-l6-h256-uncased')
model.save_pretrained('models/xtremedistil-l6-h256-uncased')