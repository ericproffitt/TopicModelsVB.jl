# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling.

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora.  Modeling this latent structure is done using either [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) methods, or [variational Bayesian](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VB) methods.  The former approach is slower, but unbiased.  Given infinite time, MCMC will fit the desired model exactly.  The latter method is faster (often much faster), but biased, since one must approximate distributions in order to ensure tractability.  This package takes the latter approach to topic modeling.

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
Let's begin with the Corpus data structure.  The Corpus data structure has been designed for maximum ease-of-use.  Datasets must still be cleaned and put into the appropriate format, but once a dataset is in the proper format and read into a corpus, it can easily be molded and modified to meet the user's needs.

There are four plaintext files that make up a corpus:
 * docfile
 * lexfile
 * userfile
 * titlefile
 
None of these files are mandatory to read a corpus, and in fact reading no files will result in an empty corpus.  However in order to train a model, a docfile will be necessary, since it contains all quantitative data known about the documents in the corpus.  On the other hand, the three remaining files are used solely for interpreting output.

The docfile should be a plaintext file containing lines of delimited numerical values.  Each document is a block of lines, the number of which depends on what information is known about the documents.  Since a document is at its essence a list of terms, each document *must* contain at least one line containing a nonempty list of delimited positive integer values corresponding to the terms from which it is composed.  Any further lines in a document block are optional, however if they are present they must be present for all documents and must come in the following order:

* ```terms```: A line of delimited positive integers corresponding to the terms which make up the document (this line is mandatory).

* ```counts```: A line of delimited positive integers equal in length to the term line, corresponding to the number of times a particular term appears in a document (defaults to ```ones(length(terms))```).

* ```readers```: A line of delimited positive integers corresponding to those users which have read the document.

* ```ratings```: A line of delimited positive integers equal in length to the ```readers``` line, corresponding to the rating each reader gave the document (defaults to ```ones(length(readers))```).

* ```stamp```: A numerical value in the range ```[-inf, inf]``` denoting the timestamp of the document.

The lex and userfiles are dictionaries mapping positive integers to terms and usernames (resp.).  For example,

```
1    this
2    is
3    a
4    lex
5    file
```

A userfile is identitcal to a lexfile, except usernames will appear in place of a vocabulary terms.

Finally, a titlefile is simply a list of titles, not a dictionary, and is of the form

```
title1
title2
title3
title4
title5
```

The order of these titles correspond to the order of document blocks in the associated docfile.

To read a corpus into TopicModelsVB.jl, use the following function:

```julia
readcorp(;docfile="", lexfile="", userfile="", titlefil="", delim=',', counts=false, readers=false, ratings=false, stamps=false)
```

The ```file``` keyword arguments indicate the path where the file is located.

It is often the case that even once files are correctly formatted and read, the corpus will still contain formatting defects which prevent the corpus from being loaded into a topic model.  Therefore, before loading a corpus into a model, it's **very important** that one of the following is run:

```julia
fixcorp!(corp; kwargs...)
```
or
```julia
padcorp!(corp; kwargs...)
fixcorp!(corp; kwargs...)
```

Padding a corpus before fixing it will ensure that any documents which contain lex or userkeys not in the lex or user dictionaries are not removed.  Instead, generic lex and userkeys will be added as necessary to the lex and user dicionaries (resp.).

**Important:** A corpus is only a container for documents.  

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order lex keys, etc.), this will not affect any corpus attached to a model.  However!  Since corpora are containers for their documents, modifying an individual document will affect this document in all corpora which contain it.  **Be very careful whenever modifying the internals of documents themselves, either manually or through the use of** ```corp!``` **functions**. 

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

nsflda = LDA(nsfcorp, 9)
train!(nsflda, iter=150, tol=0.0) # Setting tol=0.0 will ensure that all 150 iterations are completed.
                                  # If you don't want to watch the ∆elbo, set chkelbo=151.

# training...

showtopics(nsflda, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4        topic 5       topic 6      topic 7          topic 8         topic 9
data            research        species          research       research      cell         research         theory          chemistry
project         study           research         systems        university    protein      project          problems        research
research        experimental    plant            system         support       cells        data             study           metal
study           high            study            design         students      proteins     study            research        reactions
earthquake      systems         populations      data           program       gene         economic         equations       chemical
ocean           theoretical     genetic          algorithms     science       plant        important        work            study
water           phase           plants           based          scientists    genes        social           investigator    studies
studies         flow            evolutionary     control        award         studies      understanding    geometry        program
measurements    physics         population       project        dr            molecular    information      project         organic
field           quantum         data             computer       project       research     work             principal       structure
provide         materials       dr               performance    scientific    specific     development      algebraic       molecular
time            properties      studies          parallel       sciences      function     theory           mathematical    dr
models          temperature     patterns         techniques     conference    system       provide          differential    compounds
results         model           relationships    problems       national      study        analysis         groups          surface
program         dynamics        determine        models         projects      important    policy           space           molecules
```

One thing we notice is that despite producing what are clearly coherent topics, many of the top words in each topic are words such as *research*, *study*, *data*, etc.  While such terms would be considered informative in a generic corpus, they are effectively stop words in a corpus composed of science article abstracts.  Such corpus-specific stop words will be missed by most generic stop word lists, and can be a difficult to pinpoint and individually remove.  Thus let's change our model to a filtered latent Dirichlet allocation (fLDA) model.
```julia
srand(1)

nsfflda = fLDA(nsfcorp, 9)
train!(nsfflda, iter=150, tol=0.0)

# training...

showtopics(nsfflda, 20, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4           topic 5          topic 6       topic 7          topic 8         topic 9
earthquake      theoretical     species          algorithms        university       cell          economic         theory          chemistry
ocean           physics         plant            parallel          students         protein       social           equations       reactions
water           flow            genetic          performance       program          cells         theory           geometry        chemical
measurements    phase           populations      computer          science          plant         policy           mathematical    metal
program         quantum         evolutionary     processing        scientists       proteins      human            differential    program
soil            particle        plants           applications      sciences         gene          change           algebraic       molecular
climate         temperature     population       network           scientific       genes         political        groups          organic
seismic         phenomena       patterns         networks          conference       molecular     public           solutions       surface
global          energy          variation        software          national         function      science          mathematics     compounds
sea             measurements    dna              computational     projects         expression    decision         finite          molecules
response        laser           ecology          efficient         engineering      regulation    people           dimensional     electron
earth           particles       food             distributed       year             plants        labor            spaces          university
solar           numerical       test             program           workshop         dna           market           functions       reaction
pacific         liquid          ecological       power             months           mechanisms    scientific       manifolds       synthesis
damage          fluid           host             programming       mathematical     membrane      factors          professor       spectroscopy
surface         heat            diversity        estimation        faculty          cellular      environmental    applications    complexes
samples         waves           forest           implementation    institute        genetic       theoretical      operators       species
ground          theory          environmental    programs          equipment        binding       cultural         nonlinear       energy
atmospheric     transport       animals          dynamic           international    brain         women            algebra         professor
ice             optical         reproductive     neural            nsf              enzymes       relationship     boundary        carbon

```

We can now see that many of the most troublesome corpus-specific stop words have been automatically filtered out, while those that remain are mostly those which tend to cluster within their own, more generic, topic.

### CTM
For our final example using the NSF Abstracts corpus, let's upgrade our model to a filtered *correlated* topic model (fCTM)
```julia
srand(1)

nsffctm = fCTM(nsfcorp, 9)
train!(nsffctm, iter=150, tol=0.0)

# training...

showtopics(nsffctm, 20, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4           topic 5         topic 6       topic 7        topic 8         topic 9
earthquake      flow            species          design            university      protein       social         theory          chemistry
ocean           experimental    plant            algorithms        support         cell          economic       equations       chemical
water           materials       genetic          models            students        cells         theory         investigator    reactions
program         model           populations      parallel          program         proteins      policy         geometry        metal
measurements    phase           plants           computer          science         gene          models         mathematical    molecular
soil            theoretical     evolutionary     performance       scientists      plant         change         differential    program
models          optical         population       model             award           genes         human          algebraic       dr
climate         temperature     dr               processing        dr              molecular     public         groups          properties
seismic         particle        patterns         applications      scientific      dr            model          space           organic
global          models          evolution        network           sciences        regulation    political      solutions       university
sea             heat            relationships    networks          conference      plants        examine        mathematics     surface
effects         properties      dna              software          national        expression    case           spaces          electron
response        growth          variation        efficient         projects        mechanisms    issues         dimensional     molecules
pacific         fluid           effects          computational     engineering     dna           people         finite          compounds
earth           numerical       biology          distributed       year            membrane      theoretical    functions       reaction
solar           surface         molecular        programming       researchers     growth        effects        questions       synthesis
model           quantum         reproductive     estimation        workshop        binding       factors        manifolds       spectroscopy
atmospheric     effects         animals          program           months          acid          decision       properties      energy
damage          laser           growth           implementation    mathematical    enzymes       labor          professor       dynamics
change          phenomena       test             algorithm         faculty         site          market         operators       materials
```

Because the topics in the fLDA were already so well defined, there's little room for improvement in topic coherence by upgrading to the fCTM model, however what's most interesting about the CTM and fCTM models is the ability to look at correlations between topics.

Based on the top 20 terms in each topic, we might tentatively assign the following topic labels:

* topic 1: *Earth Science*
* topic 2: *Physics*
* topic 3: *Sociobiology*
* topic 4: *Computer Science*
* topic 5: *Academia*
* topic 6: *Microbiology*
* topic 7: *Economics*
* topic 8: *Mathematics*
* topic 9: *Chemistry*

Now let's take a look at the topic-covariance matrix
```julia
model.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
model.sigma[4,8] # 15.005
model.sigma[3,6] # 13.219
model.sigma[2,9] # 7.502

# Top 3 negative entries, sorted in ascending order:
model.sigma[6,8] # -22.347
model.sigma[3,8] # -20.198
model.sigma[4,6] # -14.160
```

According to the list above, the most closely related topics are topics 4 and 8, which correspond to the *Computer Science* and *Mathematics* topics, followed closely by 3 and 6, corresponding to the topics *Sociobiology* and *Microbiology*, and then by 2 and 9, corresponding to *Physics* and *Mathematics*.

As for the least associated topics, the most unrelated pair of topics is 6 and 8, corresponding to *Microbiology* and *Mathematics*, followed closely by topics 3 and 8, corresponding to *Sociobiology* and *Mathematics*, and then third are topics 4 and 6, corresponding to *Computer Science* and *Microbiology*.

Interestingly, the topic which is least correlated with all other topics is not the *Academia* topic (which is the second least correlated), but the *Economics* topic
```julia
sum(abs(model.sigma[:,7])) - model.sigma[7,7] # Economics topic, absolute off-diagonal covariance 5.732.
sum(abs(model.sigma[:,5])) - model.sigma[5,5] # Academia topic, absolute off-diagonal covariance 18.766.
```

Looking closer at ```model.sigma```, it appears that there is a tendency within the natural sciences for the softer side of the spectrum to use slightly more academic buzzwords, while the harder sciences tend to eschew them.  The *Economics* topic is also the only non-natural science found among the 9 topics, and thus its lack of overlapping lexicon with the natural sciences likely leads to little correlation between between it and the remaining 8 topics.

### DTM
Now that we have covered static topic models, let's transition to the dynamic topic model (DTM).  The dynamic topic model discovers the temporal-dynamics of topics which, nevertheless, remain thematically static.  A good example of a topic which is thematically-static, yet exhibits an evolving lexicon, is computer storage.  Methods of data storage have evolved rapidly in the last 40 years.  Evolving from punch cards, to 5-inch floppy disks, to smaller hard disks, to zip drives and cds, to dvds and platter hard drives, and now to flash drives, solid-state drives and cloud storage, all accompanied by the rise and fall of computer companies which manufacture (or at one time manufactured) these products.

As our example, let's consider a corpus of approximately 8000 Apple magazine articles, drawn from the magazines *MacWorld* and *MacAddict*, published between the years 1984 - 2005.  We sample 400 articles randomly from each year.
```julia
srand(1)

cmagcorp = readcorp(:cmag)

cmagcorp.docs = filter(doc -> doc.title[1:3] == "Mac", cmagcorp.docs)
cmagcorp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, cmagcorp.docs), 400, replace=false) for y in 1984:2005]...)

fixcorp!(corp, stop=true, order=false, b=200, len=10)

cmaglda = fLDA(corp, 8)
train!(cmagflda, iter=150, chkelbo=151)

# training...

cmagdtm = DTM(cmagcorp, 8, 200, cmagflda)
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
Now let's evaluate the accuracy of this model against the test set.  Where the baseline for the mean accuracy will be ```mean(acc) = 0.5```.
```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc)
```

# Parallel Computing Support

# Types
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


# Functions
### Generic Functions
```julia
isnegative(.)
# Take a number or an array of numbers and return Bool or Array{Bool} (resp.).

ispositive(.)
# Take a number or an array of numbers and return Bool or Array{Bool} (resp.).

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
# Verify that all Document fields have legal values.

checkcorp(corp::Corpus)
# Verify that all Corpus fields have legal values.

readcorp(;docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Read corpus from plaintext files.

writecorp(corp::Corpus; docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Write corpus to plaintext files.

abridgecorp!(corp::Corpus; stop::Bool=false, order::Bool=true, b::Int=1)
# Abridge corpus.
# If stop = true, stop words are removed.
# If order = false, order is ignored and multiple seperate occurrences of words are stacked and the associated counts increased.
# All terms which appear < b times are removed from documents.

trimcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true)
# Those values which appear in the indicated fields of documents, yet don't appear in the corpus dictionaries, are removed.

compactcorp!(corp::Corpus; lex::Bool=true, users::Bool=true, alphabet::Bool=true)
# Compact a corpus by relabeling lex and/or userkeys so that they form a unit range.
# If alphabet=true the lex and/or user dictionaries are alphabetized.

padcorp!(corp::Corpus; lex::Bool=true, users::Bool=true)
# Pad a corpus by entering generic values for lex and/or userkeys which appear in documents but not in the lex/user dictionaries.

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
# Display the text and title of a document(s).

getlex(corp::Corpus)
# Collect sorted values from the lex dictionary.

getusers(corp::Corpus)
# Collect sorted values from the user dictionary.
```

### Model Functions
```julia
checkmodel(model::TopicModel)
# Verify that all model fields have legal values.

train!(model::Union{LDA, fLDA, CTM, fCTM}; iter::Int=200, tol::Float64=1.0, niter=1000, ntol::Float64=1/model.K^2, viter::Int=10, vtol::Float64=1/model.K^2, chkelbo::Int=1)
# Train one of the following models: LDA, fLDA, CTM, fCTM.
# 'iter'    - the maximum number of iterations through the corpus
# 'tol'     - the absolute tolerance and ∆elbo required as a stopping criterion.
# 'niter'   - the maximum number of iterations for Newton's and interior-point Newton's methods.
# 'ntol'    - the tolerance for the change of function value as a stopping criterion for Newton's and interior-point Newton's methods.
# 'viter'   - the maximum number of iterations for optimizing the variational parameters (at the document level).
# 'vtol'    - the tolerance for the change of variational parameter values as a stopping criterion.
# 'chkelbo' - how often the elbo should be checked (for both user evaluation and convergence).

train!(dtm::DTM; iter::Int=200, tol::Float64=1.0, niter=1000, ntol::Float64=1/dtm.K^2, cgiter::Int=100, cgtol::Float64=1/dtm.T^2, chkelbo::Int=1)
# Train DTM.
# 'cgiter' - the maximum number of iterations for the Polak-Ribière conjugate gradient method.
# 'cgtol'  - the tolerance for the change of function value as a stopping criterion for Polak-Ribière conjugate gradient method.

train!(ctpf::CTPF; iter::Int=200, tol::Float64=1.0, viter::Int=10, vtol::Float64=1/ctpf.K^2, chkelbo::Int=1)
# Train CTPF.

gendoc(model::Union{LDA, fLDA, CTM, fCTM}, a::Real=0.0)
# Generate a generic document from the model parameters by running the associated graphical model as a generative process.
# The argument 'a' uses Laplace smoothing to smooth the topic-term distributions.

gencorp(model::Union{LDA, fLDA, CTM, fCTM}, corpsize::Int, a::Real=0.0)
# Generate a generic corpus of size 'corpsize' from the model parameters.

showtopics(model::TopicModel, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), cols::Int=4)
# Display the top 'N' words for each topic in 'topics', defaults to 4 columns per line.

showtopics(dtm::DTM, N::Int=min(15, dtm.V); topics::Union{Int, Vector{Int}}=collect(1:dtm.K), times::Union{Int, Vector{Int}}=collect(1:dtm.T), cols::Int=4)
# Display the top 'N' words for each topic in 'topics' and each time interval in 'times', defaults to 4 columns per line.

showlibs(ctpf::CTPF, users::Union{Int, Vector{Int}})
# Show the document(s) in a user's library.

showdrecs(ctpf::CTPF, docs::Union{Int, Vector{Int}}, U::Int=min(16, ctpf.U); cols::Int=4)
# Show the top 'U' user recommendations for a document(s), defaults to 4 columns per line.

showurecs(ctpf::CTPF, users::Union{Int, Vector{Int}}=Int[], M::Int=min(10, ctpf.M); cols::Int=1)
# Show the top 'M' document recommendations for a user(s), defaults to 1 column per line.
# If a document has no title, the documents index in the corpus will be shown instead.

```

# Bibliography
1. Latent Dirichlet Allocation (2003); Blei, Ng, Jordan. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)
2. Correlated Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006.pdf)
3. Dynamic Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf)
4. Content-based Recommendations with Poisson Factorization (2014); Gopalan, Charlin, Blei. [pdf](http://www.cs.columbia.edu/~blei/papers/GopalanCharlinBlei2014.pdf)
5. Numerical Optimization (2006); Nocedal, Wright. [Amazon](https://www.amazon.com/Numerical-Optimization-Operations-Financial-Engineering/dp/0387303030)
6. Machine Learning: A Probabilistic Perspective (2012); Murphy. [Amazon](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr=)
