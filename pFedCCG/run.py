import random
import numpy as np
import torch
from pFedCCG_recsys import pFedCCG_recsys
from pFedCCG_classification import pFedCCG_classification
from pFedCCG_sft import pFedCCG_SFT
from config import get_args

if __name__ == '__main__':
    args, cfg = get_args()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    if args.task == "classification":
        pFedCCG = pFedCCG_classification

        args.dataset = "cifar10"
        args.partition = "noniid"
        args.lr = 1e-2
        args.epochs = 10

    elif args.task == "recsys":
        pFedCCG = pFedCCG_recsys

        args.dataset = "ml1m"
        args.recsys_model = "ncf"
        args.lr = 1e-2
        args.epochs = 5

    elif args.task == "llm_sft":
        pFedCCG = pFedCCG_SFT

    pfedccg = pFedCCG(args, cfg)

    for now_time in range(args.comm_round):
        pfedccg.step(now_time)





