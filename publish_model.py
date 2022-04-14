from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_name_or_path="../sbert_xtremedistil-l6-h256-uncased_mean_cosine")

# Adjust repo name to reflect model
model.save_to_hub(repo_name='acm_challenge', private=True, commit_message='First trained Model')