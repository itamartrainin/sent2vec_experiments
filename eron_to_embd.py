import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()


dataset_dir = r'C:\Users\Itamar Trainin\Documents\Datasets\ENRON mail\maildir'
data_dir = r'C:\Data\sent2vec_experiments'


emails_df = pd.read_csv(data_dir + '/subjects.csv')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def sent2vec(text):
    if not text or type(text) != str:
        return None
    return model.encode([text])


vecs = []
for one in tqdm(emails_df['subjects'], total=len(emails_df['subjects'])):
    vec = sent2vec(one)
    vecs.append(vec)

vecs = np.array(vecs)
np.save(data_dir + '/embeds', vecs)

# emails_df['embeddings'] = emails_df['subjects'].progress_apply(sent2vec)
# emails_df.to_csv(data_dir + '/subjects_embeds.csv')


