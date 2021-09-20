abbert2: Language Models for Antibodies
=======================================

Welcome to abbert2 :wave:, a library to enable training of machine learning
models over large repertoires of antibodies.

**abbert2 is young and under heavy development, so expect rough edges**

Getting started
---------------

```shell
# clone the repo
git clone https://github.com/bayer-science-for-a-better-life/abbert-cerebras.git abbert2
cd abbert2

# create the conda environment
conda create -f environment.yml

# activate it
conda activate abbert2

# if everything worked well, this should work
# it is a small example of how to access the preprocessed data dumps from OAS
python examples/oas101.py
```


OAS
---

The [Observed Antibody Space](http://opig.stats.ox.ac.uk/webapps/oas/) (OAS) data
is licensed under a [CC-BY 4.0 license](https://creativecommons.org/licences/by/4.0/).
