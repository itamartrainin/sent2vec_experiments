import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

data_dir = r'C:\Data\sent2vec_experiments'

k = 20

emails_df = pd.read_csv(data_dir + '/subjects_embeds.csv')
emails_df_nonan = emails_df[~emails_df['embeddings'].isna()]

vecs = np.array(emails_df_nonan['embeddings'])

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(vecs)
distances, indices = nbrs.kneighbors(vecs)