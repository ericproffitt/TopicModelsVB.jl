# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora of documents.  Modeling this latent thematic structure is done using either Markov chain Monte Carlo methods, or variational Bayesian methods.  The former approach is slower but unbiased, in that given infinite time, the desired model will be fit exactly.  The latter method is faster (often much faster), but is biased.  This package takes the latter approach to topic modeling.
