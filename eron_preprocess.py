import os
import re
import string
import pandas as pd
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

tqdm.pandas()

dataset_dir = r'C:\Users\Itamar Trainin\Documents\Datasets\ENRON mail\maildir'
data_dir = r'C:\Data\sent2vec_experiments'

files_counter = 0
for subdir, dirs, files in os.walk(dataset_dir):
    if os.path.basename(subdir) == 'all_documents':
        for file in files:
            files_counter += 1

texts = []
with tqdm(total=files_counter) as bar:
    for subdir, dirs, files in os.walk(dataset_dir):
        if os.path.basename(subdir) == 'all_documents':
            for file in files:
                bar.update(1)
                with open(os.path.join(subdir, file), 'r') as f:
                    try:
                        text = f.read()
                    except:
                        continue
                    texts.append([os.path.join(subdir, file), text])

print('Done reading the emails.')

emails_df = pd.DataFrame(texts, columns=['paths', 'texts'])
print('Total files processed: {}'.format(len(emails_df)))


def extract_subject(text):
    text = text.split('Subject: ')
    text = 'Subject: '.join(text[1:])

    end = re.search(r'[a-zA-Z \-]+: ', text)
    end = end.span()[0]

    subject = text[:end]
    subject = subject.strip()
    subject = re.sub(r'\s+', ' ', subject)

    if subject:
        return subject
    else:
        return ''


emails_df['subjects'] = emails_df['texts'].progress_apply(extract_subject)
print('Total subjects extracted: {}'.format(len(emails_df[emails_df['subjects'] != ''])))


# Subject cleaning
ps = PorterStemmer()
stopWords = set(stopwords.words('english'))


def stem_subject(text):
    print(text)
    if not text:
        return ''
    tokens = word_tokenize(text)
    tokens = map(lambda word: ps.stem(word), tokens)
    tokens = filter(lambda word: word not in stopWords and word not in string.punctuation, tokens)
    tokens = list(tokens)
    tokens = ' '.join(tokens)
    print(tokens)
    return tokens


emails_df['cleaned'] = emails_df['subjects'].progress_apply(stem_subject)


# emails_df = pd.read_csv(data_dir + '/subjects.csv')
emails_df.to_csv(data_dir + '/subjects.csv')
