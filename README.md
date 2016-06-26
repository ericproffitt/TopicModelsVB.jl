# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling.

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora.  Modeling this latent structure is done using either Markov chain Monte Carlo (MCMC) methods, or variational Bayesian (VB) methods.  The former approach is slower, but unbiased.  Given infinite time, MCMC will fit the desired model exactly.  The latter method is faster (often much faster), but biased, since one must approximate distributions in order to insure tractability.  This package takes the latter approach to topic modeling.

# Dependencies
```julia
Pkg.add("Distributions.jl")
```

# Install
```julia
Pkg.add("TopicModelsVB")
```

# Datasets
Included in TopicModelsVB.jl are three datasets:

1. National Science Foundation Abstracts 1989 - 2003:
  * 128804 documents
  * 25319 lexicon

2. CiteULike Science Article Database:
  * 16980 documents
  * 8000 lexicon
  * 5551 users

3. Computer Magazine Archive Article Collection 1975 - 2014:
  * 330577 documents
  * 16020 lexicon

# Corpus
Let's begin with the Corpus data structure.  The Corpus data structure has been designed for maximum ease-of-use.  Datasets must still be cleaned and put into the appropriate format, but once a dataset is in the proper format, corpora can be easily molded and modified to meet the user's needs.

Any useful corpus needs a non-empty collection of documents.  The document file should be a plaintext file containing lines of delimited numerical values.  Each document is a block of lines, the number of which depends on the amount of information one has about the document.  Since a document is essential a list of terms, each document *must* contain at least one line containing a list of delimited positive integer values corresponding to the terms from which it is composed.  The lines for a particular document block (if they are present) must come in the following order:

1. This line is mandatory, and is a delimited list of positive integers corresponding to the terms which make up the document.

2. A line of delimited positive integers equal in length to the first line, corresponding to the number of times a particular term appears in a document.

3. A line delimited positive integers corresponding to the readers which have the corresponding document in their library.

4. A line of delimited positive integers equal in length to the third line, corresponding to the rating each reader gave the corresponding document.

5. A numerical value in the range ```[-inf, inf]``` denoting the timestamp of the document.

```julia
readcorp(;docfile, lexfile, userfile, titlefile, delim::Char, counts::Bool, readers::Bool, ratings::Bool, stamps::Bool)
```

The file keywords are all strings indicating the path where the file is located.  It's not necessary to include all (or even any) of the files.  Loading no files will simply return an empty corpus.

Even once the files are correctly formatted and read into a corpus, it's often still the case that the corpus is not sufficiently cleaned and formatted to be usable by the models.  Thus before loading a corpus into a model, it's **very important** that the user always runs one of the following
```julia
fixcorp!(corp; kwargs...)
```
or
```julia
padcorp!(corp; kwargs...)
fixcorp!(corp; kwargs...)
```

Padding a corpus before fixing it will insure that any documents which contain lexkeys or userkeys not in the lex or user dictionaries attached to the corpus are not removed.  Instead, generic lex and userkeys will be added to the lex and user dicionaries (resp.).

**Important:** The Corpus type is just a container for documents, along with two dictionaries which give lex and user keys sensible names.  

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order lex keys, etc.), this will not affect any corpus attached to a model.  However!  Since corpora are containers for their documents, modifying an individual document will affect this document in all corpora which contain it.  **Be very careful whenever you modify the internals of documents themselves, either manually or through the use of** ```corp``` **functions**. 

# Models
The available models are as follows:
```julia
LDA(corp, K)
# Latent Dirichlet Allocation model with K topics.

fLDA(corp, K)
# Filtered latent Dirichlet allocation model with K topics.

CTM(corp, K)
# Correlated topic model with K topics.

fCTM(corp, K)
# Filtered correlated topic model with K topics.

DTM(corp, K, delta, pmodel)
# Dynamic topic model with K topics and ∆ = delta.

CTPF(corp, K, pmodel)
# Collaborative topic Poisson factorization model with K topics.
```

Notice that both ```DTM``` and ```CTPF``` have a ```pmodel``` argument.  It is **highly advisable** that you prime these final two models with a pretrained model from one of the first four, otherwise learning may take a prohibitively long time.

# Tutorial
### LDA
Let's begin our tutorial with a simple latent Dirichlet allocation (LDA) model with 8 topics, trained on the first 5000 documents from the NSF Abstracts corpus.
```julia
using TopicModelsVB

srand(1)

nsfcorp = readcorp(:nsf)
nsfcorp.docs = nsfcorp[1:5000]
fixcorp!(nsfcorp)

# Notice that the post-fix lexicon is considerably smaller after removing all but the first 5000 docs.

nsflda = LDA(nsfcorp, 8)
train!(nsflda, iter=200)

# training...

showtopics(nsflda)
```

One thing we notice is all the words which would be considered informative to a generic corpus, but which are effectively stop words in a corpus of science article abstracts.  These words will be missed by most stop word lists, and can be a pain to pinpoint and individually remove.  Thus let's change our model to a filtered latent Dirichlet allocation (fLDA) model.
```julia
srand(1)

nsfflda = fLDA(nsfcorp, 8)
train!(nsfflda, iter=200)

# training...

showtopics(nsfflda)
```

Now we see can see that many of the most troublesome corpus-specific stop words have been automatically filtered out of the topics, and those that remain are those which tend to belong to their own, more generic, topic.

### CTM
For our final test with the NSF Abstracts corpus, let's upgrade our model to a filtered *correlated* topic model (fCTM)
```julia
srand(1)

nsffctm = fCTM(nsfcorp, 8)
train!(nsffctm, iter=200)

# training...

showtopics(nsffctm, 20)
```
Not only have corpus-specific stop words been removed, but we can see that the topics are significantly more well defined and consistent than in the non-correlated LDA model.

Based on the top 20 terms in each topic, we might tentatively assign the following topic labels:

• topic 1: *Earth Science*

• topic 2: *Physics*

• topic 3: *Microbiology*

• topic 4: *Computer Science*

• topic 5: *Academia*

• topic 6: *Sociobiology*

• topic 7: *Chemistry*

• topic 8: *Mathematics*

Now let's take a look at the topic-covariance matrix
```julia
model.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
model.sigma[4,8] # 16.770
model.sigma[3,6] # 14.758
model.sigma[2,8] # 8.970

# Top 3 negative entries, sorted in ascending order:
model.sigma[6,8] # -21.677
model.sigma[1,8] # -19.016
model.sigma[2,6] # -15.107
```

According to the list above, the most closely related topics are topics 4 and 8, which correspond to the *Computer Science* and *Mathematics* topics, followed closely by 3 and 6, corresponding to the topics *Microbiology* and *Sociobiology*, and then by 2 and 8, corresponding to *Physics* and *Mathematics*.

As for the least associated topics, the most unrelated pair of topics is 6 and 8, corresponding to *Sociobiology* and *Mathematics*, followed closely by topics 1 and 8, corresponding to *Earth Science* and *Mathematics*, and then third are topics 2 and 6, corresponding to *Physics* and *Sociobiology*.

Interestingly enough, the topic which is least correlated with all other topics is not in fact the *Academia* topic (which is the second least correlated), but the *Chemistry* topic
```julia
sum(abs(model.sigma[:,7])) - model.sigma[7,7] # Chemistry topic, absolute off-diagonal covariance 0.037.
sum(abs(model.sigma[:,5])) - model.sigma[5,5] # Academia topic, absolute off-diagonal covariance 16.904.
```
however looking at the variance of these two topics
```julia
model.sigma[7,7] # 0.002
model.sigma[5,5] # 16.675
```
it appears suprisingly that the *Chemistry* topic appears to use a very tightly controlled and idiosyncratic lexicon, with few vocabulary words shared across topics, unlike the way topics such as *Physics* and *Mathematics*, or *Microbiology* and *Sociobiology* tend to share terminology.

### DTM
Now that we have covered static topic models, let's transition to the dynamic topic model (DTM).  The dynamic topic model looks lexical temporal-dynamics of topics which are, nevertheless, thematically static.  A good example a topic which is thematically-static, but which exhibits an evolving lexicon, is computer storage.  

Methods of data storage have evolved rapidly in the last 40 years.  Evolving from punch cards, to 5-inch floppy disks, to smaller hard disks, to zip drives and cds, to dvds and platter hard drives, and now to flash drives, solid-state drives and cloud storage, all accompanied by the rise and fall of computer companies which manufacture (or at one time manufactured) these products.

For our example, let's take a look at 11,000 Apple magazine articles, drawn from *MacWorld* and *MacAddict* magazine, between the years 1984 - 2005, where we sample 500 articles randomly from each year.
```julia
srand(1)

cmagcorp = readcorp(:cmag)

cmagcorp.docs = filter(doc -> doc.title[1:3] == "Mac", cmagcorp.docs)
cmagcorp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, cmagcorp.docs), 500, replace=false) for y in 1984:2005]...)

fixcorp!(corp, stop=true, order=false, b=150, len=10)

cmaglda = fLDA(corp, 10)
train!(cmagflda, iter=150, chkelbo=151)

# training...

cmagdtm = DTM(cmagcorp, 10, 200, cmagflda)
train!(cmagdtm, cgiter=10, iter=200)

# training...

showtopics(model, 20, topics=5)
```

### CTPF
Finally, we take a look at a topic model which is not primarily interested in the topics, but rather in their ability to collaborative filtering in order to better recommend users unseen documents.  The collaborative toipc Poisson fatorization (CTPF) model blends the latent thematic structure of documents with the document-user matrix, in order to obtain higher accuracy than would be achievable with just the user library information, and also overcomes the cold-start problem for documents with no readers.  Let's take the CiteULike dataset and randomly remove a single reader from each of the documents
```julia
srand(1)

citeucorp = readcorp(:citeu)

testukeys = Int[]
for doc in citeucorp
    index = sample(1:length(doc.readers), 1)
    push!(testukeys, doc.readers[index])
    deleteat!(doc.readers, index)
end

fixcorp!(citeucorp)
```

Notice that 158 of the the documents had only a single reader (no documents had 0 readers), since CTPF can depend entirely on thematic structure for making recommendations if need be, this poses no problem for the model.

Now let's train a ```CTPF``` model on our modified corpus, and then we will evaluate the success of our model at imputing the correct users back into document libraries
```julia
citeulda = LDA(citeucorp, 8)
train!(citeulda, iter=150)

# training...

citeuctpf = CTPF(citeucorp, 8, citeulda)
train!(citeuctpf, iter=200)

# training...
```
Now let's evaluate the accuracy of this model against the test set.  Where the base line is ```mean(acc) = 0.5```.
```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc)
```

# Miscellaneous Material

# Type Master-list
```julia
VectorList
# Array{Array{T,1},1}

MatrixList
# Array{Array{T,2},1}

Document(terms; counts=ones(length(terms)), readers=Int[], ratings=ones(length(readers)), stamp=-Inf, title="")
# FIELDNAMES:
# terms::Vector{Int}
# counts::Vector{Int}
# readers::Vector{Int}
# ratings::Vector{Int}
# stamp::Float64
# title::UTF8String

Corpus(;docs=Document[], lex=[], users=[])
# FIELDNAMES:
# docs::Vector{Document}
# lex::Dict{Int, UTF8String}
# users::Dict{Int, UTF8String}

TopicModel
# abstract type

LDA(corp, K) <: TopicModel
# Latent Dirichlet allocation
# 'K' denotes the number of topics.

fLDA(corp, K) <: TopicModel
# Filtered latent Dirichlet allocation

CTM(corp, K) <: TopicModel
# Correlated topic model

fCTM(corp, K) <: TopicModel
# Filtered correlated topic model

DTM(corp, K, delta, pmodel) <: TopicModel
# Dynamic topic model
# 'delta' denotes the time-step size.
# 'pmodel' is a pre-trained model from Union{LDA, fLDA, CTM, fCTM}

CTPF(corp, K, pmodel) <: TopicModel
# Collaborative topic Poisson factorization
```


# Function Master-list
### Generic Functions
```julia
isnegative(.)
# Takes a number or an array of numbers and returns Bool or Array{Bool} (resp.).

ispositive(.)
# Takes a number or an array of numbers and returns Bool or Array{Bool} (resp.).

tetragamma(.)
# polygamma(2, x)

partition(xs, n)
# xs: Vector or UnitRange
# n: positive Int

# Returns a VectorList containing contiguous portions of xs of length n (includes remainder).
# e.g. partition([1,5,"HI",5,-7.1], 2) == Vector[[1,5],["HI",5],[-7.1]]
```

### Document/Corpus Functions
```julia
checkdoc(doc::Document)
# Verifies that all Document fields have legal values.

checkcorp(corp::Corpus)
# Verifies that all Corpus fields have legal values.

readcorp(;docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Reads corpus from plaintext files.

writecorp(corp::Corpus; docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Writes a corpus to plaintext files.

abridgecorp!(corp::Corpus; stop::Bool=false, order::Bool=true, b::Int=1)
# Abridges a corpus.
# If stop = true, stop words are removed.
# If order = false, order is ignored and multiple seperate occurrences of words are stacked and the associated counts increased.
# All terms which appear < b times are removed from documents.

trimcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true)
# Those values which appear in the indicated fields of documents, but don't appear in the corpus dictionaries, are removed.

compactcorp!(corp::Corpus; lex::Bool=true, users::Bool=true, alphabet::Bool=true)
# Compacts a corpus by relabeling lex and/or user keys so that they form a unit range.
# If alphabet=true, the lex and/or user dictionaries are alphabetized.

padcorp!(corp::Corpus; lex::Bool=true, users::Bool=true)
# Pads a corpus by entering generic values for lex and/or user keys which appear in documents but not in the corpus dictionaries.

cullcorp!(corp::Corpus; terms::Bool=false, readers::Bool=false, len::Int=1)
# Culls the corpus of documents which contain lex and/or user keys in a document's terms/readers (resp.) fields yet don't appear in the corpus dictionaries.
# All documents of length < len are removed.

fixcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true, stop::Bool=false, order::Bool=true, b::Int=1, len::Int=1, alphabet::Bool=true)
# Fixes a corp by running the following four functions:
# abridgecorp!(corp, stop=stop, order=order, b=b)
# trimcorp!(corp, lex=lex, terms=terms, users=users, readers=readers)
# cullcorp!(corp, len=len)	
# compactcorp!(corp, lex=lex, users=users, alphabet=alphabet)

showdocs(corp::Corpus, docs::Union{Document, Vector{Document}, Int, Vector{Int}})
# Displays the text and title of a document.

getlex(corp::Corpus)
# Collects the sorted values from the corpus lexicon

getusers(corp::Corpus)
# Collects the sorted values from the list of corpus users.
```

### Model Functions
```julia
checkmodel(model::TopicModel)
# Verifies that all model fields have legal values.

train!(model::Union{LDA, fLDA, CTM, fCTM}; iter::Int=200, tol::Float64=1.0, niter=1000, ntol::Float64=1/model.K^2, viter::Int=10, vtol::Float64=1/model.K^2, chkelbo::Int=1)
# Trains one of the models: LDA, fLDA, CTM, fCTM.
# 'iter'    - the maximum number of iterations through the corpus
# 'tol'     - the absolute tolerance and ∆elbo required as a stopping criterion.
# 'niter'   - the maximum number of iterations for Newton's and interior-point Newton's methods.
# 'ntol'    - the tolerance for the change of function value as a stopping criterion for Newton's and interior-point Newton's methods.
# 'viter'   - the maximum number of iterations for optimizing the variational parameters (at the document level).
# 'vtol'    - the tolerance for the change of variational parameter values as a stopping criterion.
# 'chkelbo' - how often the elbo should be checked (for both user evaluation and convergence).

train!(dtm::DTM; iter::Int=200, tol::Float64=1.0, niter=1000, ntol::Float64=1/dtm.K^2, cgiter::Int=100, cgtol::Float64=1/dtm.T^2, chkelbo::Int=1)
# Trains DTM.
# 'cgiter' - the maximum number of iterations for the Polak-Ribière conjugate gradient method.
# 'cgtol'  - the tolerance for the change of function value as a stopping criterion for Polak-Ribière conjugate gradient method.

train!(ctpf::CTPF; iter::Int=200, tol::Float64=1.0, viter::Int=10, vtol::Float64=1/ctpf.K^2, chkelbo::Int=1)
# Trains CTPF.

gendoc(model::Union{LDA, fLDA, CTM, fCTM}, a::Real=0.0)
# Generates a generic document from the model parameters by running the associated graphical model as a generative process.
# The argument 'a' uses Laplace smoothing to smooth the topic-term distribution.

gencorp(model::Union{LDA, fLDA, CTM, fCTM}, corpsize::Int, a::Real=0.0)
# Generates a generic corpus of size 'corpsize' from the model parameters by running gendoc(corp, a) iteratively.

showtopics(model::TopicModel, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), cols::Int=4)
# Displays the top 'N' words for each topic in 'topics', defaults to 4 columns per line.

showtopics(dtm::DTM, N::Int=min(15, dtm.V); topics::Union{Int, Vector{Int}}=collect(1:dtm.K), times::Union{Int, Vector{Int}}=collect(1:dtm.T), cols::Int=4)
# Displays the top 'N' words for each topic in 'topics' and each time interval in 'times', defaults to 4 columns per line.

showlibs(ctpf::CTPF, users::Union{Int, Vector{Int}})
# Shows the documents in a user's library.

showdrecs(ctpf::CTPF, docs::Union{Int, Vector{Int}}, U::Int=min(16, ctpf.U); cols::Int=4)
# Shows the top 'U' users that are recommended to read a document(s), defaults to 4 columns per line.

showurecs(ctpf::CTPF, users::Union{Int, Vector{Int}}=Int[], M::Int=min(10, ctpf.M); cols::Int=1)
# Shows the top 'M' documents recommnded for a user(s), defaults to 1 column per line.
# If a document has no title, the documents index in the corpus will be shown instead.

```

# Advanced Material
###Setting Hyperparameters
###DTM Variations
###Hidden Markov Topic Model (HMTM)

# Bibliography

