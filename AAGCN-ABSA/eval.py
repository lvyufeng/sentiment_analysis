from ast import Num
import os
import time

from mindspore import load_checkpoint, load_param_into_net
from mindspore import context

from data_utils import build_dataset
from models.aagcn import AAGCN
from train import parse_args
import mindspore.ops as ops
import mindspore
from mindspore import context, Tensor

import mindspore.nn as nn

from train import EvalEngine

def test_eval():
    opt = parse_args()
    context.set_context(
        mode=context.GRAPH_MODE, 
        device_target=opt.device, 
        device_id=opt.device_id,
    )
    if opt.save_graphs:
        context.set_context(save_graphs=True, save_graphs_path="./_save_{}".format(int(time.time())))
    begin = time.time()
    _, val_dataset, test_dataset, embedding_matrix = build_dataset(
        dataset_prefix=opt.dataset_prefix, 
        knowledge_base=opt.knowledge_base,
        worker_num=opt.worker_num,
        valset_ratio = opt.valset_ratio
    )
    
    val_dataset = test_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    t2 = time.time()
    print('load data:',t2-begin)
    # import numpy
    # embedding_matrix = numpy.load('data/com_embedding_matrix.npy', allow_pickle=True)
    net = AAGCN(embedding_matrix, opt)

    # import torch
    # torch_model = torch.load('weight/aagcn_15_rest_senticnet.pkl')
    # keys = torch_model.keys()
    # m2t_map = {
    #     'embed.embedding_table':'embed.weight'
    # }
    # for p in net.get_parameters():
    #     if p.name in keys:
    #         data = torch_model[p.name].numpy()
    #     elif p.name in m2t_map:
    #         data = torch_model[m2t_map[p.name]].numpy()
    #     else:
    #         print(p.name)
    #     p.set_data(Tensor(data, dtype=p.dtype))

    if os.path.exists(opt.pretrained):
        param_dict = load_checkpoint(opt.pretrained)
        load_param_into_net(net, param_dict)
        for m in net.get_parameters():
            print(m)

    engine = EvalEngine(net)
    print(engine.eval(val_dataset))

if __name__=='__main__':
    test_eval()