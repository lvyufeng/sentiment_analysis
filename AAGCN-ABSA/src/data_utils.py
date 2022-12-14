# -*- coding: utf-8 -*-
import os
import pickle
from random import shuffle
import mindspore
import numpy as np
import mindspore.dataset.engine as de
print('split shuffle')
def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = './dataset/embedding_matrix/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('[ARGS] loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('[ARGS] loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('[ARGS] building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            # map idx-word word-idx
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

def get_tokenizer(dataset_prefix, dataset_paths):
    if os.path.exists('./dataset/word2idx/'+dataset_prefix+'_word2idx.pkl'):
        print("[ARGS] loading {0} tokenizer...".format(dataset_prefix))
        with open('./dataset/word2idx/'+dataset_prefix+'_word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
            tokenizer = Tokenizer(word2idx=word2idx)
    else:
        print('build new tokenizer...')
        text = ''
        for dataset_path in dataset_paths:
            with open(dataset_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                try:
                    entity, attribute = lines[i + 1].lower().strip().split()
                except:
                    entity = lines[i + 1].lower().strip()
                    attribute = ''
                text += text_raw + " " + entity + " " + attribute + " "
        tokenizer = Tokenizer()
        tokenizer.fit_on_text(text)
        with open('./dataset/word2idx/'+dataset_prefix+'_word2idx.pkl', 'wb') as f:
            pickle.dump(tokenizer.word2idx, f)
    return tokenizer

def padding(datas, mode:str ='max', content_length:int=80, grade_size:int = 32, pad_key:str='text_indices'):
    '''
        padding for different datas' shape
        Args:
            datas(tensor): origin datas
            mode(str): padding mode
                max:   padding all datas' length to global max value
                fix:   padding all datas' length to fix value, need to give content_length as fix value
                grade: padding all datas' legnth to different grade, need to give grade_size as per grade size
            content_length:  fix value in fix mode
            grade_size: per grade size in grade mode

            pad_key: pad data depend column pad_key
    '''
    if mode=='grade':
        content_length_list = []
        for data in datas:
            grade = len(data['text_indices'])//grade_size
            grade_len = (grade+1)*grade_size
            content_length_list.append(grade_len)
    else:
        if mode=='max':
            print('padding mode:',mode)
            content_length = max([len(data[pad_key]) for data in datas])
        content_length_list = [content_length]*len(datas)
    for idx, data in enumerate(datas):
        seq_length = len(data[pad_key])
        if content_length_list[idx]>len(data[pad_key]):
            pad_length = content_length_list[idx]-len(data[pad_key])
            pad_data = datas[idx][pad_key] + [0]*pad_length
            entity_graph = np.pad(
                data['entity_graph'],
                ((0,pad_length),(0,pad_length)), 
                'constant')
            attribute_graph = np.pad(
                data['attribute_graph'],
                ((0,pad_length),(0,pad_length)), 
                'constant')
        else:
            pad_data = data[pad_key][:content_length_list[idx]]
            entity_graph = data['entity_graph'][:content_length_list[idx]]
            attribute_graph = data['attribute_graph'][:content_length_list[idx]]
        datas[idx] = (
            pad_data, 
            data['polarity'], 
            entity_graph, 
            attribute_graph, 
            seq_length
        )
    return datas

class ABSADataSet():
    def __init__(self, dataset_path='', dataset_prefix='', embed_dim=300, tokenize=None):
        self.dataset_path = dataset_path
        self.embed_dim = embed_dim
        self.dataset_prefix = dataset_prefix
        self.dataset = self._read_data(self.dataset_path, tokenize)

    def _read_data(self, fname, tokenizer):
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            lines = f.readlines()
        with open(fname + '.graph_entity', 'rb') as f:
            entity_graphs = pickle.load(f)
        with open(fname + '.graph_attribute', 'rb') as f:
            attribute_graphs = pickle.load(f)
        all_data = []
        graph_id = 0

        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            polarity = lines[i+2].strip()
            text_indices = tokenizer.text_to_sequence(text)
            try:
                polarity = int(polarity)+1
            except:
                print(text)
            entity_graph = entity_graphs[graph_id]
            attribute_graph = attribute_graphs[graph_id]
            all_data.append((text_indices, polarity, entity_graph, attribute_graph, len(text_indices)))
            graph_id += 1

        return all_data
    
    # ????????????
    def __getitem__(self, index):
        return self.dataset[index]

    # ????????????
    def __len__(self):
        return len(self.dataset)

def build_ABSA(dataset='', data_keys=[], batch_size=16, worker_num=1, shuffle=True):
    dataset = de.GeneratorDataset(
        dataset,
        data_keys,
        shuffle=shuffle,
        num_parallel_workers=worker_num,
    )
    dataset = dataset.bucket_batch_by_length(
        column_names=['text_indices'], bucket_boundaries=[16, 24, 32],
        bucket_batch_sizes= [batch_size] * 4, drop_remainder=False
    )
    return dataset

def accumulate(lengths):
    offsets = [0]
    for idx, i in enumerate(lengths):
        offsets.append(i + offsets[idx])
    return offsets[1:]

def random_split(dataset, lengths):
    shuffle(dataset)
    return [dataset[offset - length: offset] for offset, length in zip(accumulate(lengths), lengths)]

def build_dataset(dataset_prefix, knowledge_base, batch_size=16, worker_num=1, valset_ratio=0):
    train_dataset_path = './dataset/{}/{}_train.raw.tokenized'.format(knowledge_base, dataset_prefix)
    test_dataset_path = './dataset/{}/{}_test.raw.tokenized'.format(knowledge_base, dataset_prefix)
    tokenize = get_tokenizer(dataset_prefix, [train_dataset_path, test_dataset_path])
    embedding_matrix = build_embedding_matrix(tokenize.word2idx, 300, dataset_prefix)
    data_keys = ['text_indices', 'polarity', 'entity_graph', 'attribute_graph', 'seq_length']

    train_dataset = ABSADataSet(
        dataset_path=train_dataset_path,
        dataset_prefix=dataset_prefix,
        embed_dim=300,
        tokenize=tokenize
    ).dataset

    test_dataset = ABSADataSet(
        dataset_path=test_dataset_path,
        dataset_prefix=dataset_prefix,
        embed_dim=300,
        tokenize=tokenize
    ).dataset

    if valset_ratio>0:
        valset_len = int(len(train_dataset) * valset_ratio)
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - valset_len, valset_len))
    else:
        val_dataset = test_dataset

    train_loader = build_ABSA(
        dataset=train_dataset, data_keys=data_keys, 
        batch_size=batch_size, worker_num=worker_num, shuffle=False)
    val_loader = build_ABSA(
        dataset=val_dataset, data_keys=data_keys,
        batch_size=batch_size, worker_num=worker_num, shuffle=False)
    test_loader = build_ABSA(
        dataset=test_dataset, data_keys=data_keys, 
        batch_size=batch_size, worker_num=worker_num, shuffle=False)
    return train_loader, val_loader, test_loader, embedding_matrix

if __name__=='__main__':
    build_dataset(
        dataset_prefix='15_rest', 
        knowledge_base='senticnet',
        worker_num=1,
        valset_ratio = 0.1
    )