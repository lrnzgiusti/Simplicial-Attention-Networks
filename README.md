# Simplicial Attention Neural Networks (SAN)

This is the official code for the paper:
[Simplicial Attention Neural Networks](https://arxiv.org/abs/2203.07485), *arXiv:2203.07485*, 2022.

### Imputing missing data with graph neural networks

We present simplicial attention neural networks (SNNs), a generalization of graph attention networks to data defined over simplicial complexes.


### Organization of the code

The core of the layers described in the paper can be found in *layers/simplicial_attention_layer.py*. The file *test.py* runs the imputation over the citation complex with specification given as parameters to the script. Table 2 can be reproduced by executing *run.sh*.

### References

[1] Veličković, Petar, et al. **Graph Attention Networks**. arXiv preprint arXiv:1710.10903 (2017).
[2] Kipf, T.N. and Welling, M., 2016. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907.
[4] Ebli, Stefania, Michaël Defferrard, and Gard Spreemann. **Simplicial neural networks**. arXiv preprint arXiv:2010.03633 (2020).
[3] Barbarossa, Sergio, and Stefania Sardellitti. **Topological signal processing over simplicial complexes**. IEEE Transactions on Signal Processing 68 (2020): 2992-3007.

### Cite

Please cite [our paper](https://arxiv.org/abs/2203.07485) if you use this code in your own work:
