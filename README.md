# TADA  

 ## instruction
 TADA, an efficient and effective front-mounted data augmentation framework for GNNs on HDGs. Under the hood, TADA includes two key modules: (i) feature expansion with structure embeddings, and (ii) topology- and attribute-aware graph sparsification. The former obtains augmented node features and enhanced model capacity by encoding the graph structure into high-quality structure embeddings with our highly-efficient sketching method. Further, by exploiting taskrelevant features extracted from graph structures and attributes, the second module enables the accurate identification and reduction of numerous redundant/noisy edges from the input graph, thereby alleviating over-smoothing and facilitating faster feature aggregations over HDGs. Empirically, TADA considerably improves the predictive performance of mainstream GNN models on 8 real homophilic/heterophilic HDGs in terms of node classification, while achieving efficient training and inference processes.

## Environment settings

- python==3.11.5
- pytorch==2.0.1
- cuda==12.1
- torch_geometric==2.4.0

you can create and activate the environments by following code :

    conda env create -f environments.yml
    conda activate tada
    

## Run

if you want to reproduce the results of GNNs and GNNs+TADA., please run the following command:

    bash run.sh




## Acknowledgment
Our code is based on the official code of [Dspar](https://github.com/zirui-ray-liu/DSpar_tmlr) and [MGNN](https://github.com/GuanyuCui/MGNN/tree/main/src)
.
