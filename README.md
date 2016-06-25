# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora of documents.  Modeling this latent thematic structure is done using either Markov chain Monte Carlo methods, or variational Bayesian methods.  The former approach is slower but unbiased, in that given infinite time, the desired model will be fit exactly.  The latter method is faster (often much faster), but is biased.  This package takes the latter approach to topic modeling.

# Datasets
Included in TopicModelsVB.jl are three datasets:

1. The National Science Foundation Abstracts 1989 - 2003
  * 128804 documents
  * 25319 lexicon

2. The CiteULike Science Article database
  * 16980 documents
  * 8000 lexicon
  * 5551 users

3. Computer Magazine Archive articles 1975 - 2014
  * 330577 documents
  * 16020 lexicon

# Simple Tutorial
```julia
using TopicModelsVB
```
