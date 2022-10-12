import torch
import argparse
import warnings
from utils.load import load_acm, load_dblp, load_freebase, load_imdb
from utils.evaluate import evaluate
from module.model import MAE
from torch_geometric import seed_everything
import yaml
seed_everything(65536)
warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def train(args):
    if args.dataset == "Acm":
        load_data = load_acm
    elif args.dataset == "DBLP":
        load_data = load_dblp
    elif args.dataset == "Freebase":
        load_data = load_freebase
    elif args.dataset == "IMDB":
        load_data = load_imdb

    data = load_data().to(device)

    model = MAE(
        data,
        args.dim,
        args.p1,
        args.p2,
        args.rate,
        args.noise,
        1,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.w)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

    model.eval()
    embeds = model.get_embeds(data)
    for ratio in args.ratio:
        evaluate(
            embeds,
            ratio,
            data[data.main_node][f'{ratio}_train_mask'],
            data[data.main_node][f'{ratio}_val_mask'],
            data[data.main_node][f'{ratio}_test_mask'],
            data[data.main_node].y,
            device,
            args.dataset,
            args.lr1,
            0.0
        )


parser = argparse.ArgumentParser(description="HeMAE")
parser.add_argument("--dataset", type=str, default="Acm")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--ratio", type=int, default=[20, 40, 60])
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
train(args)
