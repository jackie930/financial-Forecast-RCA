"""
1. Clean data 
2. form data for bert-single and bert-user-info
3. form data for bert-cnn and bert-lstm
4. generate pseudo user info 
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class RandomEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, avg_embedding_norm=1):
        super().__init__(0, 0)
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.avg_embedding_norm = avg_embedding_norm
        n, d = self.num_embeddings, self.embedding_dim
        b = int(np.ceil(n/d))
        self.weight = None
        rand_weight = torch.randn(b, d)
        curr_avg_norm = (d * torch.sum(rand_weight[:b-1,:].norm(dim=1)) + 
                        (n - (b-1) * d) * rand_weight[b-1,:].norm()) / n
        rand_weight *= avg_embedding_norm / curr_avg_norm.item()
        ind = torch.arange(d)
        rand_signs = torch.randint(2, (b,d), dtype=torch.bool)
        self.register_buffer('rand_weight', rand_weight)
        self.register_buffer('ind', ind)
        self.register_buffer('rand_signs', rand_signs)

    def forward(self, input):
        if input.dtype != torch.long:
            raise TypeError('Input must be of type torch.long')
        if (torch.sum(input >= self.num_embeddings).item() != 0 or 
                torch.sum(input < 0).item() != 0):
            raise ValueError('Entries of input tensor must all be non-negative '
                             'integers less than self.num_embeddings')
        d = self.embedding_dim
        input_us = input.unsqueeze(-1)
        return (self.rand_weight[torch.div(input_us, d, rounding_mode='floor'),  (input_us - self.ind) % d] * 
               (self.rand_signs[torch.div(input, d, rounding_mode='floor'), :] * 2.0 - 1.0))



def remove_punctaution(st):
    st = st.translate(str.maketrans(' ',' ', '!"#$%&\'()*+-./:;<=>?@[\\]^_`{|}~'))
    return st

def remove_digital(st):
    new_st = ''
    for char in st:
        if not char.isdigit():
            new_st = new_st + char
        else:
            new_st = new_st + ''
    return new_st

def remove_punc(st):
    punctuations = '''!()-[]{|};:'"\,<>./?@#$%^&*_`~=+'''
    new_st = ""
    for char in st:
        if char not in punctuations:
            new_st = new_st + char
        else:
            new_st = new_st + ' '

    return new_st


def clean_data(input_path, data_name):
    df = pd.read_csv(input_path+data_name, sep=',')
    df.fillna(' ', inplace=True)
    df['industry'] = df['industry_name_sc'].apply(remove_punctaution)
    df['concept'] = df['concept_name_sc'].apply(remove_punctaution)
    
    # combine industru y and concept as tags
    df['tags'] = df['industry'] + ',' + df['concept']
    df['tags'] = df['tags'].apply(lambda x: x.replace(',', ' '))


    # remove puncaution
    df['clean_content'] = df['content'].apply(remove_punc)
    df['clean_title'] = df['title'].apply(remove_punc)
    df['clean_abstract'] = df['abstract'].apply(remove_punc)


    # remove digital
    df['clean_content'] = df['clean_content'].apply(remove_digital)
    df['clean_title'] = df['clean_title'].apply(remove_digital)
    df['clean_abstract'] = df['clean_abstract'].apply(remove_digital)

    # save clean data
    clean_data_name = 'clean_df.csv'
    df.to_csv(input_path+clean_data_name, index=False)
#     print('clean data is saved in ', input_path+clean_data_name)
    return df


def create_sequential_data(input_path, data_name):
    df = clean_data(input_path, data_name)
    # sort data records according to event time
    sort_df = df.sort_values(['event_time'],ascending=True).groupby('user_no')
    # keep the most 3 records of each user
    top_df = sort_df.head(3).reset_index()[['user_no', 'event_time', 'key_label', 'clean_title', 'clean_abstract', 'clean_content', 'tags']]
    # define the input which is the combination of columns '[clean_title, clean_abstract, clean_content, tags]'
    top_df['input_info'] = top_df['clean_title'] +' '+top_df['clean_abstract']+' ' + top_df['clean_content']

    new_df = pd.DataFrame(columns=['user_no', 'key_label', 'input_1', 'input_2', 'input_3'])
    new_df['user_no'] = top_df['user_no'].unique()
    unique_users = top_df['user_no'].unique()

    labels = []
    for u in unique_users:
        label = top_df[top_df.user_no==u].key_label.unique()[0]
        labels.append(label)

    new_df['key_label'] = labels

    input_1 = []
    input_2 = []
    input_3 = []
    for u in unique_users:
        user_df = top_df[top_df['user_no'] == u][['user_no', 'input_info']]
        input_1.append(user_df[user_df['user_no']==u]['input_info'].values[0])
        try: 
            input_2.append(user_df[user_df['user_no']==u]['input_info'].values[1])
        except: 
            input_2.append(' ') # padding
        try: 
            input_3.append(user_df[user_df['user_no']==u]['input_info'].values[2])
        except: 
            input_3.append(' ') # padding


    new_df['input_1'] = input_1
    new_df['input_2'] = input_2
    new_df['input_3'] = input_3
    
    new_df_name = 'user_top_3_data.csv'
    new_df.to_csv(input_path+new_df_name, index=False)
#     print('data is saved in ', input_path+new_df_name)
    return new_df 

    
    


# # prepare data from BERT-CNN and BERT-LSTM

# # sort data records according to event time
# sort_df = df.sort_values(['event_time'],ascending=True).groupby('user_no')

# # keep the most 3 records of each user
# top_df = sort_df.head(3).reset_index()[['user_no', 'event_time', 'key_label', 'clean_title', 'clean_abstract', 'clean_content', 'tags']]
# # define the input which is the combination of columns '[clean_title, clean_abstract, clean_content, tags]'
# top_df['input_info'] = top_df['clean_title'] +' '+top_df['clean_abstract']+' ' + top_df['clean_content']

# new_df = pd.DataFrame(columns=['user_no', 'key_label', 'input_1', 'input_2', 'input_3'])
# new_df['user_no'] = top_df['user_no'].unique()
# unique_users = top_df['user_no'].unique()

# labels = []
# for u in unique_users:
#     label = top_df[top_df.user_no==u].key_label.unique()[0]
#     labels.append(label)
    
# new_df['key_label'] = labels

# input_1 = []
# input_2 = []
# input_3 = []
# for u in unique_users:
#     user_df = top_df[top_df['user_no'] == u][['user_no', 'input_info']]
#     input_1.append(user_df[user_df['user_no']==u]['input_info'].values[0])
#     try: 
#         input_2.append(user_df[user_df['user_no']==u]['input_info'].values[1])
#     except: 
#         input_2.append(' ') # padding
#     try: 
#         input_3.append(user_df[user_df['user_no']==u]['input_info'].values[2])
#     except: 
#         input_3.append(' ') # padding
        

# new_df['input_1'] = input_1
# new_df['input_2'] = input_2
# new_df['input_3'] = input_3

# new_df.to_csv(input_path+'user_top_3_data.csv', index=False)


# generate pseudo user info 

# user_number = df['user_no'].nunique()
# user_emb_size = 256
# emb = RandomEmbedding(user_number,user_emb_size,avg_embedding_norm=1)

# user_ids = torch.tensor(list(range(user_number)), dtype=torch.int64)
# user_embeddings = emb(user_ids)
# user_embeddings.shape

def generate_pseudo_user_embeddings(user_num, emb_dimension):
    emb = RandomEmbedding(user_num, emb_dimension, avg_embedding_norm=1)
    user_ids = torch.tensor(list(range(user_num)), dtype=torch.int64)
    user_embeddings = emb(user_ids)
    return user_embeddings
