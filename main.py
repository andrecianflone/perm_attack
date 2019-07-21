import json, os, sys
import datetime
# Comet will timeout if no internet
try:
    from comet_ml import Experiment
except Exception as e:
    from comet_ml import OfflineExperiment
from types import MethodType
# from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import optim
import utils, models, text_attacks
# from torchtext import datasets
# from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss
# from torch.nn.modules.loss import MultiLabelMarginLoss, BCELoss
from attack_models import Seq2Seq, PermSeq2Seq
import dataHelper
import hparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):

    # Load data
    train_loader,test_loader,dev_loader =\
                                    utils.get_data(args, args.prepared_data)

    # The unknown model to attack, specified in args.model
    unk_model = utils.load_unk_model(args,train_loader,test_loader)

    ntokens = args.vocab_size

    # Load model which will produce the attack
    G = PermSeq2Seq(emsize=args.emsize,
                          glove_weights=args.embeddings,
                          train_emb=args.train_emb,
                          nhidden=args.nhidden,
                          ntokens=ntokens,
                          nlayers=args.nlayers,
                          noise_radius=args.noise_radius,
                          hidden_init=args.hidden_init,
                          dropout=args.dropout,
                          deterministic=args.deterministic_G).to(device)

    # Parallel compute
    if args.data_parallel:
        G = nn.DataParallel(G)

    # Load saved
    if args.load_model:
        G = torch.load(args.adv_model_path)
        print("Loaded saved model from: {}".format(args.adv_model_path))

    # Opt and Loss
    opt = optim.Adam(G.parameters())

    # Train
    text_attacks.train_white_box(args, train_loader, test_loader,
                                            dev_loader, unk_model, G, opt)

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    args = hparams.get_params()
    main(args)


