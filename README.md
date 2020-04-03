# nyu-ds1016
Group Project for NYU-DS1016, Computational Cognitive Modeling (2020 Spring Semester)

## Group Members
* Wangrui (Wendy) Hou  |  wh916
* Gabriella (Gaby) Hurtado  |  gh1408
* Stephen Roy  |  sr5388

## Topic: Neural Networks - Language
_Selected from [CCM Project Site](https://brendenlake.github.io/CCM-site/final_project_ideas.html)_  
>Exploring lexical and grammatical structure in BERT or GPT2. What do powerful pre-trained models learn about lexical and grammatical structure? Explore the learned representations of a state-of-the-art language model (BERT, GPT2, etc.) in systematic ways, and discuss the learned representation in relation to how children may acquire this structure through learning.

## Timeline
_Adpated from [CCM Site](https://brendenlake.github.io/CCM-site/#final-project)_
1. Initial Meeting (2020-03-26)
2. Proposal Review (2020-04-02)
3. Proposal Submission (2020-04-06)
  * The final project proposal is due Monday, April 6 (one half page written). Please submit via email to instructors-ccm-spring2020@nyuccl.org with the file name lastname1-lastname2-lastname3-ccm-proposal.pdf.
  * https://piazza.com/class/k5cskqm4l1d4ei?cid=87
4. Final Project Due (2020-05-13)

## Dataset sources: 
* TBD

## Working Documents
* [Google Drive](https://drive.google.com/drive/u/2/folders/1f8UW4vlJ13Tse6Tcq5q4cHt_pFy9068-)
* [Proposal - Draft](https://drive.google.com/open?id=1LsgqYxx-ldeHlZdwNQ3P9O6NL-R3YThtyOMT-5xseJs)

## TODO
- [ ] @stephen, set up git
- [ ] @stephen, select data set
- [ ] @stephen, bert integration with project
- [ ] @wendy, BERT history
- [ ] @wendy, template proposal
- [ ] @gaby, NL tasks/tests/history


## Acknowledgements:
* [Ganesh Jawahar - intpret_bert upstream repo](https://ganeshjawahar.github.io/)
* [Ganesh speaking about interpret_bert](https://vimeo.com/384961703)
* [Children First Language Acquisition At Age 1-3 Years Old In Balata](http://www.iosrjournals.org/iosr-jhss/papers/Vol20-issue8/Version-5/F020855157.pdf)
* [Caregivers' Role in Child's Language Acquisition](https://dspace.univ-adrar.dz/jspui/handle/123456789/2476)
* [The Acquisition of Syntax](https://linguistics.ucla.edu/people/hyams/28%20Hyams-Orfitelli.final.pdf)
* [Studies in Child Language: An Anthropological View: A First Language: The Early Stages](https://www.researchgate.net/publication/249422499_Studies_in_Child_Language_An_Anthropological_View_A_First_Language_The_Early_Stages_Roger_Brown_Language_Acquisition_and_Communicative_Choice_Susan_Ervin-Tripp_Studies_of_Child_Language_Development_Ch)
* [Stanford WordBank Dataset](http://wordbank.stanford.edu/analyses)
* [A Structural Probe for Finding Syntax in Word Representations](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf)

# Below is the original markdown from Ganesh's upstream repo README.md

---

## What does BERT learn about the structure of language?

Code used in our [ACL'19 paper](https://drive.google.com/open?id=166ngGwApN5XdOnUzs_y12GqdDCoPvUeh) for interpreting [BERT model](https://arxiv.org/abs/1810.04805).

### Dependencies
* [PyTorch](https://pytorch.org/)
* [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
* [SentEval](https://github.com/facebookresearch/SentEval)
* [spaCy](https://spacy.io/) (for dependency tree visualization)

### Quick Start

#### Phrasal Syntax (Section 3 in paper)
* Navigate:
```
cd chunking/
```
* Download the train set from [CoNLL-2000 chunking corpus](https://www.clips.uantwerpen.be/conll2000/chunking/):
```
wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
gunzip train.txt.gz
```
The last command replaces `train.txt.gz` file with `train.txt` file.
* Extract BERT features for chunking related tasks (clustering and visualization):
```
python extract_features.py --train_file train.txt --output_file chunking_rep.json
```
* Run t-SNE of span embeddings for each BERT layer (Figure 1):
```
python visualize.py --feat_file chunking_rep.json --output_file_prefix tsne_layer_
```
This would create one t-SNE plot for each BERT layer and stores as pdf (e.g. `tsne_layer_0.pdf`).
* Run KMeans to evaluate the clustering performance of span embeddings for each BERT layer (Table 1):
```
python cluster.py --feat_file chunking_rep.json
```

#### Probing Tasks (Section 4)
* Navigate:
```
cd probing/
```
* Download the [data files for 10 probing tasks](https://github.com/facebookresearch/SentEval/tree/master/data/probing) (e.g. `tree_depth.txt`)
* Extract BERT features for sentence level probing tasks (`tree_depth` in this case):
```
python extract_features.py --data_file tree_depth.txt --output_file tree_depth_rep.json
```
In the above command, append `--untrained_bert` flag to extract untrained BERT features.
* Train the probing classifier for a given BERT layer (indexed from 0) and evaluate the performance (Table 2):
```
python classifier.py --labels_file tree_depth.txt --feats_file tree_depth_rep.json --layer 0
```
We use the hyperparameter search space recommended by [SentEval](https://arxiv.org/abs/1803.05449).

#### Subject-Verb Agreement (SVA) (Section 5)
* Navigate:
```
cd sva/
```
* Download the [data file for SVA task](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz) and extract it.
* Extract BERT features for SVA task:
```
python extract_features.py --data_file agr_50_mostcommon_10K.tsv --output_folder ./
``` 
* Train the binary classifier for a given BERT layer (indexed from 0) and evaluate the performance (Table 3):
```
python classifier.py --input_folder ./ --layer 0
```
We use the hyperparameter search space recommended by [SentEval](https://arxiv.org/abs/1803.05449).

#### Compositional Structure (Section 6)
* Navigate:
```
cd tpdn/
```
* Download the [SNLI 1.0 corpus](https://nlp.stanford.edu/projects/snli/) and extract it.
* Extract BERT features for premise sentences present in SNLI:
```
python extract_features.py --input_folder . --output_folder .
```
* Train the Tensor Product Decomposition Network (TPDN) to approximate a given BERT layer (indexed from 0) and evaluate the performance (Table 4):
```
python approx.py --input_folder . --output_folder . --layer 0
```
Check `--role_scheme` and `--rand_tree` flags for setting the role scheme.
* Induce dependency parse tree from attention weights for a given attention head and BERT layer (both indexed from 1) (Figure 2):
```
python induce_dep_trees.py --sentence text "The keys to the cabinet are on the table" --head_id 11 --layer_id 2 --sentence_root 6 
```

### Acknowledgements
This repository would not be possible without the efforts of the creators/maintainers of the following libraries:
* [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) from huggingface
* [SentEval](https://github.com/facebookresearch/SentEval) from facebookresearch
* [bert-syntax](https://github.com/yoavg/bert-syntax) from yoavg
* [tpdn](https://github.com/tommccoy1/tpdn) from tommccoy1
* [rnn_agreement](https://github.com/TalLinzen/rnn_agreement) from TalLinzen
* [Chu-Liu-Edmonds](https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/mst/mst.ipynb) from bastings

### License
This repository is GPL-licensed.

