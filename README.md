# Simplicial Attention Neural Networks (SAN)

This is the official code for the paper:
[Simplicial Attention Neural Networks](https://arxiv.org/abs/2203.07485), *arXiv:2203.07485*, 2022.


![Maps](https://github.com/lrnzgiusti/Simplicial-Attention-Networks/blob/main/assets/maps.jpg)
![Architecture](https://github.com/lrnzgiusti/Simplicial-Attention-Networks/blob/main/assets/arch.png)

### Abstract 

The aim of this work is to introduce Simplicial Attention Neural Networks (SANs), i.e., novel neural architectures that operate on data defined on simplicial complexes
leveraging masked self-attentional layers. Hinging on formal arguments from topological signal processing, we introduce a proper self-attention mechanism able
to process data components at different layers (e.g., nodes, edges, triangles, and so on), while learning how to weight both upper and lower neighborhoods of the given
topological domain in a task-oriented fashion. The proposed SANs generalize most of the current architectures available for processing data defined on simplicial
complexes. The proposed approach compares favorably with other methods when applied to different (inductive and transductive) tasks such as trajectory prediction
and missing data imputations in citation complexes.

### Organization of the code

The core of the layers described in the paper can be found in *layers/simplicial_attention_layer.py*. The file *test.py* runs the imputation over the citation complex with specification given as parameters to the script. Table 2 can be reproduced by executing *run.sh*.

### References

[1] Veličković, Petar, et al. **Graph Attention Networks**. arXiv preprint arXiv:1710.10903 (2017). <br>
[2] Kipf, T.N. and Welling, M., 2016. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907. <br>
[4] Ebli, Stefania, Michaël Defferrard, and Gard Spreemann. **Simplicial neural networks**. arXiv preprint arXiv:2010.03633 (2020). <br>
[3] Barbarossa, Sergio, and Stefania Sardellitti. **Topological signal processing over simplicial complexes**. IEEE Transactions on Signal Processing 68 (2020): 2992-3007.

### Cite

Please cite [our paper](https://arxiv.org/abs/2203.07485) if you use this code in your own work:
```
@article{giusti2022simplicial,
  title={Simplicial Attention Networks},
  author={Giusti, Lorenzo and Battiloro, Claudio and Di Lorenzo, Paolo and Sardellitti, Stefania and Barbarossa, Sergio},
  journal={arXiv preprint arXiv:2203.07485},
  year={2022}
}
```
