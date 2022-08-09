import pickle, os, time, random, sys
import numpy as np
import mindspore
import mindspore.nn as nn
from cybertron import BertAdam, CrossEntropyLoss
from src.trans_module import BertEncoder, TransitionModel
from src.config import Config
from src.dataset import DataLoader, PrintMsg
from src.transform import Text2ActionSequence, Text2SingleLabel
from eval import EvaluationTrans

config = Config()
base_encoder = BertEncoder(config)
trans_model = TransitionModel(config)
crossentropy = CrossEntropyLoss()

base_encoder_optimizer = base_encoder.trainable_params()
trans_optimizer = trans_model.trainable_params()
optimizer_parameters = [
        {'params': [p for p in trans_optimizer if p.ndim > 1], 'weight_decay': config.weight_decay},
        {'params': [p for p in trans_optimizer if p.ndim == 1], 'weight_decay': 0.0},
        {'params': base_encoder_optimizer, 'lr': config.base_encoder_lr},
        {'params': trans_optimizer}]

optimizer = BertAdam(optimizer_parameters,
                     lr=config.finetune_lr,
                     warmup=config.warm_up,
                     t_total=train_iter_len * config.epochs)

for split_inx in range(1, 21):
    save_path = config.datasplit_path + '/split_' + str(split_inx)
    ## If the mode == 'new', the code np.random.seed(config.seed) should be commented out, 
    ## since we use numpy to shuffle the data
    # train, valid, test, save_path = DataLoader(doc2pair, 'new', '', config)
    train, valid, test, _ = DataLoader(None, 'old', save_path, None)
    
    train_len = len(train[0])
    train_iter_len = (train_len // config.batch_size) + 1
    train_action_sequence = Text2ActionSequence(train)
    single_labels_list = Text2SingleLabel(train)

    total_batch, early_stop = 0, 0
    best_batch, best_f1 = 0, 0.0

def train_epoch(model, dataset):
    for epoch_i in range(config.epochs):
        batch_i = 0
        while batch_i * config.batch_size < train_len:
            trans_model.train()
            base_encoder.train()
            optimizer.zero_grad()
            start, end = batch_i * config.batch_size, (batch_i +1) * config.batch_size
            document_list = train[0][start: end]
            action_sequence_list = train_action_sequence[start: end]
            single_labels = single_labels_list[start: end]
            
            # forward
            pooled, clause_state_list = base_encoder(document_list)
            sgl_logits, sgl__tensor, tpl_logits, tpl_tensor = trans_model(pooled, single_labels, clause_state_list,
                                                                          action_sequence_list, 'train')
            single_loss = crossentropy(sgl_logits, sgl__tensor)
            tuple_loss = crossentropy(tpl_logits, tpl_tensor)
            loss = single_loss + tuple_loss 
            loss.backward()
            optimizer.step()
            batch_i += 1
            total_batch += 1

            if total_batch % config.showtime == 0:
                t_start = time.time()
                valid_emo_metric, valid_cse_metric, valid_pr_metric = EvaluationTrans(trans_model, base_encoder, valid, 
                                                                                      config.batch_size)
                t_end = time.time()
                if valid_pr_metric[2] > best_f1:
                    early_stop = 0
                    best_f1 = valid_pr_metric[2]
                    best_batch = total_batch                    
                    print ('*'*50 +'the performance in valid set...' + '*'*50)
                    print('valid runging time: ', (t_end - t_start))
                    PrintMsg(total_batch, valid_emo_metric, valid_cse_metric, valid_pr_metric) 
                    torch.save(base_encoder.state_dict(), save_path + config.prefix + 'bert_best.mdl')
                    torch.save(trans_model.state_dict(), save_path + config.prefix + 'trans_best.mdl')
        early_stop += 1
        if early_stop >= config.early_num or epoch_i == config.epochs-1:
            base_encoder.load_state_dict(torch.load(save_path + config.prefix + 'bert_best.mdl'))
            trans_model.load_state_dict(torch.load(save_path + config.prefix + 'trans_best.mdl'))
            print ('='*50 +'the performance in test set...' + '='*50)
            test_emo_metric, test_cse_metric, test_pr_metric = EvaluationTrans(trans_model, base_encoder, test, 
                                                                               config.batch_size)
            PrintMsg(best_batch, test_emo_metric, test_cse_metric, test_pr_metric)
            pre, rec, f1 = test_pr_metric[0], test_pr_metric[1], test_pr_metric[2]
            base_encoder_name = config.prefix + 'bertmodel_pre_' + str(pre) + '_rec_' + str(rec) + '_f1_' + str(f1)
            trans_name = config.prefix + 'transmodel_pre_' + str(pre) + '_rec_' + str(rec) + '_f1_' + str(f1)
            torch.save(base_encoder.state_dict(), save_path + base_encoder_name + '.mdl')
            torch.save(trans_model.state_dict(), save_path + trans_name + '.mdl')
            os.remove(save_path + config.prefix + 'bert_best.mdl')
            os.remove(save_path + config.prefix + 'trans_best.mdl')
            break