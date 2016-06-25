# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling.

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora of documents.  Modeling this latent thematic structure is done using either Markov chain Monte Carlo methods, or variational Bayesian methods.  The former approach is slower but unbiased, in that given infinite time, the desired model will be fit exactly.  The latter method is faster (often much faster), but is biased.  This package takes the latter approach to topic modeling.

# Datasets
Included in TopicModelsVB.jl are three datasets:

1. The National Science Foundation Abstracts 1989 - 2003:
  * 128804 documents
  * 25319 lexicon

2. The CiteULike Science Article Database:
  * 16980 documents
  * 8000 lexicon
  * 5551 users

3. Computer Magazine Archive Articles 1975 - 2014:
  * 330577 documents
  * 16020 lexicon

# Install
```julia
Pkg.add("TopicModelsVB")
```

# Tutorial
Let's begin with the Corpus-Document datastructure.  The Corpus-Document datastructure has been designed for maximum ease-of-use.  Datasets must still be cleaned and put into the appropriate format, but once a dataset is in the proper format, corpora can be easily molded and modified to meet the user's needs.

Any useful corpus needs a non-empty collection of documents.  The document file should be a plaintext file containing lines of delimited numerical values.  Each document is a block of lines, the number of which depends on the amount of information one has about the document.  Since a document is essential a list of terms, each document *must* contain at least one line containing a list of delimited numerical values corresponding to the terms from which it is composed.  The lines for a particular document block are as follows

1. This line is mandatory, and is a delimited list of positive integers corresponding to the terms which make up the document.

2. A line of delimited positive integers equal in length to the first line, corresponding to the number of times a particular term appears in a document.

3. A line delimited positive integers corresponding to the readers which have the corresponding document in their library.

4. A line of delimited positive integers equal in length to the third line, corresponding to the rating each reader gave the corresponding document.

5. A numerical value in the range ```[-inf, inf]``` denoting the timestamp of the document.
