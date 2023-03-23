## Masked AutoEncoder for Heterogeneous Graph Representation

Self-supervised graph representation learning is a key technique for graph structured data processing, especially for Web-generated graph that do not have qualified labelling information.
## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

```python
python main.py --dataset Acm
```

The `--dataset` argument should be one of [Acm, DBLP, Freebase, IMDB].

## Reference link

The code refers to the following two papers. Thank them very much for their open source work.

[Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](https://github.com/liun-online/HeCo)

[GraphMAE: Self-Supervised Masked Graph Autoencoders](https://github.com/THUDM/GraphMAE)
