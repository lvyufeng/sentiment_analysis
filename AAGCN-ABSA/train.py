# -*- coding: utf-8 -*-
import math
import numpy
import os
import time

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Model, save_checkpoint
from mindspore.common.initializer import initializer, Uniform, XavierUniform
from mindspore.train.callback import Callback
from src.data_utils import build_dataset
from src.aagcn import AAGCN
from src.tools import print_args, parse_args


class EvalEngine():
    def __init__(self, model):
        self.model = model
        self.f1 = nn.F1()
        self.concat = ops.Concat(axis=0)

    def eval(self, dataset):
        self.model.set_train(False)
        t_targets_all, t_outputs_all = None, None
        for t_sample_batched in dataset:
            t_targets = Tensor(t_sample_batched['polarity'])
            t_outputs = self.model(
                Tensor(t_sample_batched['text_indices']),
                Tensor(t_sample_batched['entity_graph']),
                Tensor(t_sample_batched['attribute_graph']),
                Tensor(t_sample_batched['seq_length']),
            )
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = self.concat((t_targets_all, t_targets))
                t_outputs_all = self.concat((t_outputs_all, t_outputs))
        n_test_correct = int(sum((t_outputs_all.argmax(-1) == t_targets_all)))
        n_test_total = len(t_outputs_all)
        acc = n_test_correct / n_test_total
        f1 = self._f1(t_outputs_all, t_targets_all)
        self.model.set_train(True)
        return acc, f1

    def _f1(self, a, b):
        self.f1.clear()
        self.f1.update(a, b)
        return self.f1.eval(average=True)

    def update_wight(self, weight_path):
        weight = mindspore.load_checkpoint(weight_path)
        mindspore.load_param_into_net(self.model, weight)


class EvalCallBack(Callback):
    def __init__(self, net, model, opt, val_dataset, test_dataset):
        self.model = model
        self.opt = opt
        self.cur_test_epoch = 1
        self.eval_engine = EvalEngine(net)
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

        self.best_save_path = 'ckpt/best/best_{}_{}_{}_eval.ckpt'.format(
            self.opt.model_name, self.opt.dataset_prefix, self.opt.knowledge_base)
        if not os.path.exists('ckpt/best'):
            os.makedirs('ckpt/best')

        self.best_step = 0
        self.best_model_acc = 0
        self.best_model_f1 = 0
        self.sum_step_spend = 0

    def val_best_model(self):
        self.eval_engine.update_wight(self.best_save_path)
        val_acc, val_f1 = self.eval_engine.eval(self.test_dataset)
        print('>>> seed : {}, model: {}, dataset: {}, knowledge_base: {}'.format(
            self.opt.seed, self.opt.model_name, self.opt.dataset_prefix, self.opt.knowledge_base))
        print('>>> save: {}'.format(self.best_save_path))
        print('>>> VAL  best_model_acc: {:4f}, best_model_f1: {:4f} best_step: {}'.format(
            self.best_model_acc*100, self.best_model_f1*100, self.best_step))
        print('>>> TEST best_model_acc: {:4f}, best_model_f1: {:4f} best_step: {}'.format(
            val_acc*100, val_f1*100, self.best_step))

    def epoch_begin(self, run_context=None):
        cb_params = run_context.original_args()
        print('[EPOCH] epoch {}/{}'.format(cb_params.cur_epoch_num,
              cb_params.epoch_num), flush=True)
        self.epoch_begin_time = time.time()

    def epoch_end(self, run_context=None):
        cb_params = run_context.original_args()
        print('[EPOCH] epoch {}/{} finished, spend: {:6f}'.format(
            cb_params.cur_epoch_num, cb_params.epoch_num,
            time.time()-self.epoch_begin_time), flush=True)
        # 训练结束后进行测试精度
        if cb_params.cur_epoch_num == cb_params.epoch_num:
            self.val_best_model()

    def step_begin(self, run_context=None):
        self.step_begin_time = time.time()

    def _get_loss(self, cb_params):
        loss = cb_params.net_outputs
        if isinstance(loss, tuple):
            if isinstance(loss[0], Tensor):
                return loss[0].asnumpy()
        if isinstance(loss, Tensor):
            return numpy.mean(loss.asnumpy())

    def step_end(self, run_context=None):
        cb_params = run_context.original_args()
        self.loss = self._get_loss(cb_params)
        self.sum_step_spend += (time.time()-self.step_begin_time)
        if cb_params.cur_step_num % self.opt.log_step == 0:
            self.cur_test_epoch += 1
            val_acc, val_f1 = self.eval_engine.eval(self.val_dataset)
            ops = '-DROP'
            if val_acc > self.best_model_acc:
                self.best_model_acc = val_acc
            if val_f1 > self.best_model_f1:
                self.best_model_f1 = val_f1
                ops = '+SAVE'
                self.best_step = cb_params.cur_step_num
                save_checkpoint(cb_params.train_network, self.best_save_path)
            print('loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}, spend: {:.4f}, {}'.format(
                self.loss, val_acc, val_f1, self.sum_step_spend/self.opt.log_step, ops), flush=True)
            self.sum_step_spend = 0


def init_parameters(net):
    for p in net.get_parameters():
        if p.requires_grad:
            if len(p.shape) > 1:
                p.set_data(initializer(XavierUniform(), p.shape, p.dtype))
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                p.set_data(initializer(Uniform(scale=stdv), p.shape, p.dtype))


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, *input):
        out = self._backbone(input[0], input[2], input[3], input[4])
        loss = self._loss_fn(out, input[1])
        return loss


if __name__ == '__main__':
    train_begin = time.time()
    # opt = init_env()
    opt = parse_args()

    # 创建数据集
    train_dataset, val_dataset, test_dataset, embedding_matrix = build_dataset(
        dataset_prefix=opt.dataset_prefix,
        knowledge_base=opt.knowledge_base,
        worker_num=opt.worker_num,
        valset_ratio=opt.valset_ratio
    )
    # val_dataset, test_dataset = test_dataset, val_dataset
    print('train_dataset:', train_dataset.get_dataset_size())
    print('val_dataset:', val_dataset.get_dataset_size())
    print('test_dataset:', test_dataset.get_dataset_size())
    step_size = train_dataset.get_dataset_size()
    test_epoch = (step_size*opt.num_epoch)//opt.log_step
    val_dataset = val_dataset.create_dict_iterator(
        num_epochs=test_epoch)
    test_dataset = test_dataset.create_dict_iterator(
        num_epochs=1)

    # 初始化网络
    net = AAGCN(embedding_matrix, opt)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt_adm = nn.Adam(net.trainable_params(),
                      opt.learning_rate, weight_decay=opt.l2reg)
    train_net = NetWithLoss(net, loss)
    scale_manager = mindspore.DynamicLossScaleManager(2 ** 24, 2, 100)
    init_parameters(net)
    print_args(net, opt)
    train_net.set_train(True)

    print('[MODE] {}, train begin: {} epoch, {} step per epoch, {} eval'.format(
        opt.mode, opt.num_epoch, step_size, test_epoch), flush=True)
    # 创建 Model 对象
    model = Model(train_net, optimizer=opt_adm, loss_scale_manager=scale_manager)
    callback = EvalCallBack(net, train_net, opt, val_dataset, test_dataset)
    # 训练
    model.train(opt.num_epoch, train_dataset,
                callbacks=callback, dataset_sink_mode=False)
    print('train over, total spend:', time.time()-train_begin)