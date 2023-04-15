
# Project on Implementing CP-GNN (Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model)

----

Reference - 

> Linhao Luo, Yixiang Fang, Xin Cao, Xiaofeng Zhang, and Wenjie Zhang. 2021. Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21). Association for Computing Machinery, New York, NY, USA, 1170–1180. DOI:https://doi.org/10.1145/3459637.3482250

## Overview

This repository presents the implementation of the paper ”Detecting Communities from Heterogeneous Graphs: A Context Path-based Graph Neural Network Model”  which
proposes an innovative approach for identifying communities in complex, heterogeneous graphs. The method utilizes a context path-based graph neural network (CP-GNN) model to effectively capture both structural and attribute information within the graph. By employing a flexible and powerful encoding scheme, the CP-GNN model is able to accurately identify community structures in the presence of diverse node and edge attributes. This repository contains the exceutable code of CP-GNN model. details the key. Furthermore, the report showcases the results of the implementation on various datasets, comparison with autors's result and case study on the effect of k-length in CP-GNN performance.

## Environments

python == 3.6

CPU: i7-1260P 

RAM: 16.0 GB

GPU: No GPU 


## Requirements

```bash
torch==1.6.0
matplotlib==2.2.3
networkx==2.4
dgl==0.4.3.post2
numpy==1.16.6
scipy==1.4.1
scikit-learn==0.21.3
```

## Config

```python
import os

config_path = os.path.dirname(__file__)
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': 'ACM', # ACM, DBLP, IMDB, AIFB
    'data_name': 'ACM.mat', # ACM.mat, DBLP.mat, IMDB.mat, AIFB.mat
    'primary_type': 'p', # p, a, m, Personen
    'task': ['CF', 'CL'],
    'K_length': 2, # Context path length K
    'resample': False, # Whether resample the training and testing dataset
    'random_seed': 123,
    'test_ratio': 0.8
}

model_config = {
    'primary_type': data_config['primary_type'],
    'auxiliary_embedding': 'non_linear',  # auxiliary embedding generating method: non_linear, linear, emb
    'K_length': data_config['K_length'],
    'embedding_dim': 128,
    'in_dim': 128,
    'out_dim': 128,
    'num_heads': 8,
    'merge': 'linear',  # Multi head Attention merge method: linear, mean, stack
    'g_agg_type': 'mean',  # Graph representation encoder: mean, sum
    'drop_out': 0.3,
    'cgnn_non_linear': True,  # Enable non linear activation function for CGNN
    'multi_attn_linear': False,  # Enable atten K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 128,
    'path_attention': False,  # Enable Context path attention
    'c_linear_out_dim': 8,
    'enable_bilinear': False,  # Enable Bilinear for context attention
    'gru': True,
    'add_init': False
}

train_config = {
    'continue': False,
    'lr': 0.05,
    'l2': 0,
    'factor': 0.2,
    'total_epoch': 10,
    'batch_size': 1024 * 20,
    'pos_num_for_each_hop': [20, 20, 20, 20, 20, 20, 20, 20, 20],
    'neg_num_for_each_hop': [3, 3, 3, 3, 3, 3, 3, 3, 3],
    'sample_workers': 8,
    'patience': 15,
    'checkpoint_path': os.path.join(config_path, 'checkpoint', data_config['dataset'])
}

evaluate_config = {
    'method': 'LR',
    'save_heat_map': True,
    'result_path': os.path.join('result', data_config['dataset']),
    'random_state': 123,
    'max_iter': 500,
    'n_jobs': 1,
}
```

## Train and Evaluate
``` bash
python main.py
```

## Dataset Used:
![Dataset](https://i.ibb.co/8DLFhYx/ds.png)


## Implementation Process:

CP-GNN Model:

![CP-GNN Model](https://i.ibb.co/Kzvr3hk/Picture2.png)

The workflow can be visualized in the following picture: 
![Implementation Process](https://i.ibb.co/fn7XrbL/image.png)
## Result:
Result for ACM Dataset:

![enter image description here](https://i.ibb.co/LzSCZwg/image.png)

For the other three dataset with epoch =10 and k-length =2, I got the following result.
![Result](https://i.ibb.co/G5TKqbt/dataset.png)
### Case study on the effect of context path length (k-length):
As K increases, the performance of CP-GNN improves. This supports the hypothesis that higher-order relationship information is crucial for community detection.

![enter image description here](https://i.ibb.co/NSWZn22/image.png)

## Code Demo
Code demo can be found in the following link -

[Demo ](https://brocku-my.sharepoint.com/:v:/r/personal/fy21cx_brocku_ca/Documents/Attachments/Code%20Demo_Farhana%20Yasmeen.mp4?csf=1&web=1&e=f0IkuC)
