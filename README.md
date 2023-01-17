# TopicModelsVB.jl

**v1.x compatible.**

A Julia package for variational Bayesian topic modeling.

Topic models are Bayesian hierarchical models designed to discover the latent low-dimensional thematic structure within corpora. Topic models are fit using either [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC), or [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VI).

Markov chain Monte Carlo methods are slow but consistent, given enough time MCMC will fit the exact model asymptotically. Contrarily, variational inference is fast but inconsistent, as one must approximate distributions in order to ensure tractability.

This package takes the latter approach to topic modeling.

## Installation

```julia
(@v1.8) pkg> add https://github.com/ericproffitt/TopicModelsVB.jl
```

## Dependencies

```julia
DelimitedFiles
SpecialFunctions
LinearAlgebra
Random
Distributions
OpenCL
Crayons
```

## Datasets
Included in TopicModelsVB.jl are two datasets:

1. National Science Foundation Abstracts 1989 - 2003:
  * 128804 documents
  * 25319 vocabulary

2. CiteULike Science Article Database:
  * 16980 documents
  * 8000 vocabulary
  * 5551 users

## Corpus
Let's begin with the Corpus data structure. The Corpus data structure has been designed for maximum ease-of-use. Datasets must still be cleaned and put into the appropriate format, but once a dataset is in the proper format and read into a corpus, it can easily be modified to meet the user's needs.

There are four plaintext files that make up a corpus:
 * docfile
 * vocabfile
 * userfile
 * titlefile
 
None of these files are mandatory to read a corpus, and in fact reading no files will result in an empty corpus. However in order to train a model a docfile will be necessary, since it contains all quantitative data known about the documents. On the other hand, the vocab, user and title files are used solely for interpreting output.

The docfile should be a plaintext file containing lines of delimited numerical values. Each document is a block of lines, the number of which depends on what information is known about the documents. Since a document is at its essence a list of terms, each document *must* contain at least one line containing a nonempty list of delimited positive integer values corresponding to the terms of which it is composed. Any further lines in a document block are optional, however if they are present they must be present for all documents and must come in the following order:

#### terms - A line of delimited positive integers corresponding to the terms which make up the document (this line is mandatory).
#### counts - A line of delimited positive integers, equal in length to terms, corresponding to the number of times a term appears in a document.
#### readers - A line of delimited positive integers corresponding to those users which have read the document.
#### ratings - A line of delimited positive integers, equal in length to readers, corresponding to the rating each reader gave the document.

An example of a single doc block from a docfile with all possible lines included,

```
...
4,10,3,100,57
1,1,2,1,3
1,9,10
1,1,5
...
```

The vocab and user files are tab delimited dictionaries mapping positive integers to terms and usernames (resp.). For example,

```
1    this
2    is
3    a
4    vocab
5    file
```

A userfile is identitcal to a vocabfile, except usernames will appear in place of vocabulary terms.

Finally, a titlefile is simply a list of titles, not a dictionary, and is of the form,

```
title1
title2
title3
title4
title5
```

The order of these titles correspond to the order of document blocks in the associated docfile.

To read a corpus into Julia, use the following function,

```julia
readcorp(;docfile="", vocabfile="", userfile="", titlefile="", delim=',', counts=false, readers=false, ratings=false)
```

The ```file``` keyword arguments indicate the path where the respective file is located.

It is often the case that even once files are correctly formatted and read, the corpus will still contain formatting defects which prevent it from being loaded into a model. Therefore, before loading a corpus into a model, it is **important** that one of the following is run,

```julia
fixcorp!(corp)
```

or

```julia
fixcorp!(corp, pad=true)
```

Padding a corpus will ensure that any documents which contain vocab or user keys not in the vocab or user dictionaries are not removed. Instead, generic vocab and user keys will be added as necessary to the vocab and user dictionaries (resp.).

The `fixcorp!` function allows for significant customization of the corpus object.

For example, let's begin by loading the CiteULike corpus,

```julia
corp = readcorp(:citeu)
```

A standard preprocessing step might involve removing stop words, removing terms which appear less than 200 times, and alphabetizing our corpus.

```julia
fixcorp!(corp, stop=true, abridge=200, alphabetize=true, trim=true)
## Generally you will also want to trim your corpus.
## Setting trim=true will remove leftover terms from the corpus vocabulary.
```

After removing stop words and abridging our corpus, the vocabulary size has gone from 8000 to 1692.

A consequence of removing so many terms from our corpus is that some documents may now by empty. We can remove these documents from our corpus with the following command,

```julia
fixcorp!(corp, remove_empty_docs=true)
```

In addition, if you would like to preserve term order in your documents, then you should refrain from condesing your corpus.

For example,

```Julia
corp = Corpus(Document(1:9), vocab=split("the quick brown fox jumped over the lazy dog"))
showdocs(corp)
```

```
 ●●● Document 1
the quick brown fox jumped over the lazy dog
```

```Julia
fixcorp!(corp, condense=true)
showdocs(corp)
```

```
 ●●● Document 1
jumped fox over the quick dog lazy brown the
````

**Important.** A corpus is only a container for documents. 

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order vocab keys, etc.), this will not affect any corpus attached to a model. However! Since corpora are containers for their documents, modifying an individual document will affect it in all corpora which contain it. Therefore,

1. Using `fixcorp!` to modify the documents of a corpus will not result in corpus defects, but will cause them also to be changed in all other corpora which contain them.

2. If you would like to make a copy of a corpus with independent documents, use `deepcopy(corp)`.

3. Manually modifying documents is dangerous, and can result in corpus defects which cannot be fixed by `fixcorp!`. It is advised that you don't do this without good reason.

## Models
The available models are as follows:

## CPU Models
```julia
LDA(corp, K)
Latent Dirichlet allocation model with K topics.

fLDA(corp, K)
Filtered latent Dirichlet allocation model with K topics.

CTM(corp, K)
Correlated topic model with K topics.

fCTM(corp, K)
Filtered correlated topic model with K topics.

CTPF(corp, K)
Collaborative topic Poisson factorization model with K topics.
```

## GPU Models
```julia
gpuLDA(corp, K)
GPU accelerated latent Dirichlet allocation model with K topics.

gpuCTM(corp, K)
GPU accelerated correlated topic model with K topics.

gpuCTPF(corp, K)
GPU accelerated collaborative topic Poisson factorization model with K topics.
```

## Tutorial
## Latent Dirichlet Allocation
Let's begin our tutorial with a simple latent Dirichlet allocation (LDA) model with 9 topics, trained on the first 5000 documents from the NSF corpus.

```julia
using TopicModelsVB
using Random
using Distributions

Random.seed!(7);

corp = readcorp(:nsf) 

corp.docs = corp[1:5000];
fixcorp!(corp, trim=true)
## It's strongly recommended that you trim your corpus when reducing its size in order to remove excess vocabulary. 

## Notice that the post-fix vocabulary is smaller after removing all but the first 5000 docs.

model = LDA(corp, 9)

train!(model, iter=150, tol=0)
## Setting tol=0 will ensure that all 150 iterations are completed.
## If you don't want to compute the ∆elbo, set checkelbo=Inf.

## training...

showtopics(model, cols=9, 20)
```

```
topic 1        topic 2        topic 3        topic 4         topic 5         topic 6          topic 7          topic 8         topic 9
research       system         data           theory          research        research         research         research        plant
problems       research       earthquake     study           university      data             project          study           cell
design         data           project        problems        support         project          study            chemistry       species
systems        systems        research       research        students        study            data             high            protein
algorithms     control        study          equations       program         ocean            social           studies         cells
parallel       time           soil           work            science         water            understanding    properties      plants
data           design         damage         investigator    award           studies          economic         chemical        studies
project        project        seismic        principal       scientists      processes        important        materials       research
based          analysis       response       project         dr              provide          information      structure       genetic
models         processing     structures     geometry        sciences        field            policy           program         gene
model          solar          sites          mathematical    projects        time             development      surface         study
system         computer       ground         systems         conference      important        work             reactions       molecular
analysis       information    analysis       differential    scientific      climate          theory           electron        proteins
techniques     high           information    algebraic       national        marine           provide          metal           dna
methods        techniques     materials      groups          engineering     models           political        experimental    dr
problem        development    provide        space           provide         measurements     science          molecular       genes
performance    models         buildings      analysis        project         sea              models           systems         important
computer       developed      results        methods         year            species          change           energy          understanding
work           based          important      solutions       researchers     understanding    scientific       project         specific
developed      image          program        finite          mathematical    global           studies          phase           determine
```

If you are interested in the raw topic distributions. For LDA and CTM models, you may access them via the matrix,

```julia
model.beta
## K x V matrix
## K = number of topics.
## V = number of vocabulary terms, ordered identically to the keys in model.corp.vocab.
```

Now that we've trained our LDA model we can, if we want, take a look at the topic proportions for individual documents.

For instance, document 1 has topic breakdown,

```julia
println(round.(topicdist(model, 1), digits=3))
## = [0.0, 0.0, 0.0, 0.0, 0.0, 0.435, 0.082, 0.0, 0.482]
```
This vector of topic weights suggests that document 1 is mostly about biology, and in fact looking at the document text confirms this observation,

```julia
showdocs(model, 1)
## Could also have done showdocs(corp, 1).
```

```
 ●●● Document 1
 ●●● CRB: Genetic Diversity of Endangered Populations of Mysticete Whales: Mitochondrial DNA and Historical Demography
commercial exploitation past hundred years great extinction variation sizes
populations prior minimal population size current permit analyses effects 
differing levels species distributions life history...
```

Just for fun, let's consider one more document (document 25),

```julia
println(round.(topicdist(model, 25), digits=3))
## = [0.0, 0.0, 0.0, 0.849, 0.0, 0.149, 0.0, 0.0, 0.0]

showdocs(model, 25)
```

```
 ●●● Document 25
 ●●● Mathematical Sciences: Nonlinear Partial Differential Equations from Hydrodynamics
work project continues mathematical research nonlinear elliptic problems arising perfect
fluid hydrodynamics emphasis analytical study propagation waves stratified media techniques
analysis partial differential equations form basis studies primary goals understand nature 
internal presence vortex rings arise density stratification due salinity temperature...
```

We see that in this case document 25 appears to be about environmental computational fluid dynamics, which corresponds precisely to topics 4 and 6.

Furthermore, if we want to, we can also generate artificial corpora by using the ```gencorp``` function.

Generating artificial corpora will in turn run the underlying probabilistic graphical model as a generative process in order to produce entirely new collections of documents, let's try it out,

```julia
Random.seed!(7);

artificial_corp = gencorp(model, 5000, laplace_smooth=1e-5)
## The laplace_smooth argument governs the amount of Laplace smoothing (defaults to 0).

artificial_model = LDA(artificial_corp, 9)
train!(artificial_model, iter=150, tol=0, checkelbo=10)

## training...

showtopics(artificial_model, cols=9)
```

```
topic 1        topic 2      topic 3          topic 4       topic 5         topic 6        topic 7        topic 8         topic 9
system         plant        research         research      research        research       project        theory          data
research       species      project          design        study           university     data           study           research
data           cell         study            problems      chemistry       support        earthquake     problems        project
systems        studies      data             algorithms    high            students       research       research        study
control        protein      social           systems       properties      program        structures     equations       water
project        cells        important        parallel      studies         science        study          work            ocean
models         genetic      economic         project       chemical        award          response       geometry        field
processing     plants       understanding    data          materials       scientists     soil           investigator    provide
high           research     policy           models        reactions       sciences       program        principal       important
analysis       molecular    information      based         program         dr             materials      mathematical    earthquake
solar          dna          development      system        phase           scientific     information    project         analysis
design         gene         work             model         structure       projects       structural     differential    effects
time           proteins     political        analysis      experimental    engineering    seismic        algebraic       studies
computer       study        provide          methods       surface         national       sites          groups          time
performance    genes        models           techniques    electron        conference     provide        systems         marine
```

## Correlated Topic Model
For our next model, let's upgrade to a (filtered) correlated topic model (fCTM).

Filtering the correlated topic model will dynamically identify and suppress stop words which would otherwise clutter up the topic distribution output.

```julia
Random.seed!(7);

model = fCTM(corp, 9)
train!(model, tol=0, checkelbo=Inf)

## training...

showtopics(model, 20, cols=9)
```

```
topic 1          topic 2           topic 3         topic 4         topic 5         topic 6        topic 7         topic 8        topic 9
algorithms       earthquake        theory          students        ocean           economic       chemistry       physics        protein
design           data              problems        science         water           social         chemical        optical        cell
parallel         soil              equations       support         sea             theory         metal           solar          cells
system           damage            geometry        university      climate         policy         reactions       high           plant
systems          species           investigator    research        marine          political      molecular       laser          species
performance      seismic           mathematical    program         measurements    market         surface         particle       gene
problems         ground            principal       sciences        data            labor          materials       quantum        genetic
network          sites             algebraic       conference      pacific         decision       organic         devices        proteins
networks         response          differential    scientific      global          women          molecules       electron       dna
control          buildings         space           scientists      atmospheric     factors        compounds       materials      plants
based            forest            groups          national        species         human          reaction        radiation      molecular
problem          hazard            solutions       projects        trace           children       flow            temperature    genes
processing       site              mathematics     workshop        ice             public         liquid          plasma         regulation
computer         san               nonlinear       year            sediment        examine        phase           particles      expression
software         national          spaces          engineering     circulation     change         electron        magnetic       function
efficient        human             finite          faculty         north           management     properties      stars          populations
programming      archaeological    problem         mathematical    flow            population     gas             energy         specific
neural           october           manifolds       months          chemical        life           experimental    waves          binding
computational    earthquakes       dimensional     academic        samples         individuals    temperature     wave           mechanisms
distributed      patterns          numerical       equipment       mantle          competition    spectroscopy    ray            evolutionary
```

Based on the top 20 terms in each topic, we might tentatively assign the following topic labels:

* topic 1: *Computer Science*
* topic 2: *Archaeology*
* topic 3: *Mathematics*
* topic 4: *Academia*
* topic 5: *Earth Science*
* topic 6: *Economics*
* topic 7: *Chemistry*
* topic 8: *Physics*
* topic 9: *Molecular Biology*

Now let's take a look at the topic-covariance matrix,

```julia
model.sigma

## Top two off-diagonal positive entries:
model.sigma[1,3] # = 18.275
model.sigma[5,9] # = 11.393

## Top two negative entries:
model.sigma[3,9] # = -27.430
model.sigma[3,5] # = -19.441
```

According to the list above, the most closely related topics are topics 1 and 3, which correspond to the *Computer Science* and *Mathematics* topics, followed by 5 and 9, corresponding to *Earth Science* and *Molecular Biology*.

As for the most unlikely topic pairings, most strongly negatively correlated are topics 3 and 9, corresponding to *Mathematics* and *Molecular Biology*, followed by topics 3 and 5, corresponding to *Mathematics* and *Earth Science*.

## Topic Prediction

The topic models so far discussed can also be used to train a classification algorithm designed to predict the topic distribution of new, unseen documents.

Let's take our 5,000 document NSF corpus, and partition it into training and test corpora,

```julia
train_corp = copy(corp)
train_corp.docs = train_corp[1:4995];

test_corp = copy(corp)
test_corp.docs = test_corp[4996:5000];
```

Now we can train our LDA model on just the training corpus, and then use that trained model to predict the topic distributions of the five documents in our test corpus,

```julia
Random.seed!(7);

train_model = LDA(train_corp, 9)
train!(train_model, checkelbo=Inf)

test_model = predict(test_corp, train_model)
```

The `predict` function works by taking in a corpus of new, unseen documents, and a trained model, and returning a new model of the same type. This new model can then be inspected directly, or using `topicdist`, in order to see the topic distribution for the documents in the test corpus.

Let's first take a look at both the topics for the trained model and the documents in our test corpus,

```julia
showtopics(train_model, cols=9, 20)
```

```
topic 1        topic 2        topic 3        topic 4         topic 5         topic 6          topic 7          topic 8         topic 9
research       system         data           theory          research        research         research         research        plant
design         research       earthquake     study           university      data             project          study           cell
problems       data           project        problems        support         project          study            chemistry       species
systems        systems        research       research        students        study            data             high            protein
algorithms     control        study          equations       program         ocean            social           studies         cells
parallel       time           soil           work            science         water            understanding    chemical        plants
data           project        damage         investigator    award           studies          economic         properties      research
based          design         seismic        principal       scientists      processes        important        materials       studies
project        analysis       response       project         dr              provide          information      structure       genetic
models         solar          structures     geometry        sciences        field            policy           program         gene
model          processing     ground         mathematical    projects        time             development      surface         study
system         information    sites          systems         conference      important        work             reactions       molecular
analysis       high           analysis       differential    scientific      climate          theory           electron        proteins
techniques     development    information    algebraic       national        marine           provide          metal           dna
methods        techniques     materials      groups          engineering     sea              political        experimental    dr
performance    computer       provide        space           provide         models           science          molecular       genes
problem        developed      buildings      analysis        project         species          models           systems         important
computer       models         program        methods         year            measurements     change           project         understanding
work           based          important      solutions       researchers     understanding    scientific       energy          specific
developed      image          results        finite          mathematical    global           studies          phase           determine
```

```julia
showtitles(corp, 4996:5000)
```

```
 • Document 4996 Decision-Making, Modeling and Forecasting Hydrometeorologic Extremes Under Climate Change
 • Document 4997 Mathematical Sciences: Representation Theory Conference, September 13-15, 1991, Eugene, Oregon
 • Document 4998 Irregularity Modeling & Plasma Line Studies at High Latitudes
 • Document 4999 Uses and Simulation of Randomness: Applications to Cryptography,Program Checking and Counting Problems.
 • Document 5000 New Possibilities for Understanding the Role of Neuromelanin
```

Now let's take a look at the predicted topic distributions for these five documents,

```julia
for d in 1:5
    println("Document ", 4995 + d, ": ", round.(topicdist(test_model, d), digits=3))
end
```

```
Document 4996: [0.372, 0.003, 0.0, 0.0, 0.001, 0.588, 0.035, 0.001, 0.0]
Document 4997: [0.0, 0.0, 0.0, 0.538, 0.385, 0.001, 0.047, 0.027, 0.001]
Document 4998: [0.0, 0.418, 0.0, 0.0, 0.001, 0.462, 0.0, 0.118, 0.0]
Document 4999: [0.46, 0.04, 0.002, 0.431, 0.031, 0.002, 0.015, 0.002, 0.016]
Document 5000: [0.0, 0.044, 0.0, 0.001, 0.001, 0.001, 0.0, 0.173, 0.78]
```

## Collaborative Topic Poisson Factorization
For our final model, we take a look at the collaborative topic Poisson factorization (CTPF) model.

CTPF is a collaborative filtering topic model which uses the latent thematic structure of documents to improve the quality of document recommendations beyond what would be possible using just the document-user matrix alone. This blending of thematic structure with known user prefrences not only improves recommendation accuracy, but also mitigates the cold-start problem of recommending to users never-before-seen documents. As an example, let's load the CiteULike dataset into a corpus and then randomly remove a single reader from each of the documents.

```julia
Random.seed!(1);

corp = readcorp(:citeu)

ukeys_test = Int[];
for doc in corp
    index = sample(1:length(doc.readers), 1)[1]
    push!(ukeys_test, doc.readers[index])
    deleteat!(doc.readers, index)
    deleteat!(doc.ratings, index)
end
```

**Important.** We refrain from fixing our corpus in this case, first because the CiteULike dataset is pre-packaged and thus pre-fixed, but more importantly, because removing user keys from documents and then fixing a corpus may result in a re-ordering of its user dictionary, which would in turn invalidate our test set.

After training, we will evaluate model quality by measuring our model's success at imputing the correct user back into each of the document libraries.

It's also worth noting that after removing a single reader from each document, 158 of the documents now have zero readers,

```julia
sum([isempty(doc.readers) for doc in corp]) # = 158
```

Fortunately, since CTPF can if need be depend entirely on thematic structure when making recommendations, this poses no problem for the model.

Now that we've set up our experiment, let's instantiate and train a CTPF model on our corpus. Furthermore, in the interest of time, we'll also go ahead and GPU accelerate it.

```julia
model = gpuCTPF(corp, 100)
train!(model, iter=50, checkelbo=Inf)

## training...
```

Finally, we evaluate the performance of our model on the test set.

```julia
ranks = Float64[];
for (d, u) in enumerate(ukeys_test)
    urank = findall(model.drecs[d] .== u)[1]
    nrlen = length(model.drecs[d])
    push!(ranks, (nrlen - urank) / (nrlen - 1))
end
```

The following histogram shows the proportional ranking of each test user within the list of recommendations for their corresponding document.

![GPU Benchmark](https://github.com/ericproffitt/TopicModelsVB.jl/blob/master/images/ctpfbar.png)

Let's also take a look at the top recommendations for a particular document,

```julia
ukeys_test[1] # = 997
ranks[1] # = 0.978

showdrecs(model, 1, 120)
```
```
 ●●● Document 1
 ●●● The metabolic world of Escherichia coli is not small
 ...
117. #user4586
118. #user5395
119. #user531
120. #user997
```

What the above output tells us is that user 997's test document placed him or her in the top 2.2% (position 120) of all non-readers.

For evaluating our model's user recommendations, let's take a more holistic approach.

Since large heterogenous libraries make the qualitative assessment of recommendations difficult, let's search for a user with a small focused library,

```julia
showlibs(model, 1741)
```

```
 ●●● User 1741
 • Region-Based Memory Management
 • A Syntactic Approach to Type Soundness
 • Imperative Functional Programming
 • The essence of functional programming
 • Representing monads
 • The marriage of effects and monads
 • A Taste of Linear Logic
 • Monad transformers and modular interpreters
 • Comprehending Monads
 • Monads for functional programming
 • Building interpreters by composing monads
 • Typed memory management via static capabilities
 • Computational Lambda-Calculus and Monads
 • Why functional programming matters
 • Tackling the Awkward Squad: monadic input/output, concurrency, exceptions, and foreign-language calls in Haskell
 • Notions of Computation and Monads
 • Recursion schemes from comonads
 • There and back again: arrows for invertible programming
 • Composing monads using coproducts
 • An Introduction to Category Theory, Category Theory Monads, and Their Relationship to Functional Programming
```
 
 The 20 articles in user 1741's library suggest that he or she is interested in programming language theory. 
 
 Now compare this with the top 25 recommendations (the top 0.15%) made by our model,
 
```julia
showurecs(model, 1741, 25)
```

```
 ●●● User 1741
1.  On Understanding Types, Data Abstraction, and Polymorphism
2.  Functional programming with bananas, lenses, envelopes and barbed wire
3.  Can programming be liberated from the von {N}eumann style? {A} functional style and its algebra of programs
4.  Monadic Parser Combinators
5.  Domain specific embedded compilers
6.  Type Classes with Functional Dependencies
7.  Theorems for Free!
8.  Scrap your boilerplate: a practical design pattern for generic programming
9.  Types, abstraction and parametric polymorphism
10. Linear types can change the world!
11. Haskell's overlooked object system
12. Lazy functional state threads
13. Functional response of a generalist insect predator to one of its prey species in the field.
14. Improving literature based discovery support by genetic knowledge integration.
15. A new notation for arrows
16. Total Functional Programming
17. Monadic Parsing in Haskell
18. Types and programming languages
19. Applicative Programming with Effects
20. Triangle: {E}ngineering a {2D} {Q}uality {M}esh {G}enerator and {D}elaunay {T}riangulator
21. Motion doodles: an interface for sketching character motion
22. 'I've Got Nothing to Hide' and Other Misunderstandings of Privacy
23. Human cis natural antisense transcripts initiated by transposable elements.
24. Codata and Comonads in Haskell
25. How to make ad-hoc polymorphism less ad hoc
```

For the CTPF models, you may access the raw topic distributions by computing,

```julia
model.alef ./ model.bet
```

Raw scores, as well as document and user recommendations, may be accessed via,

```julia
model.scores
## M x U matrix
## M = number of documents, ordered identically to the documents in model.corp.docs.
## U = number of users, ordered identically to the keys in model.corp.users.

model.drecs
model.urecs
```

Note, as was done by Blei et al. in their original paper, if you would like to warm start your CTPF model using the topic distributions generated by one of the other models, simply do the following prior to training your model,

```julia
ctpf_model.alef = exp.(model.beta)
## For model of type: LDA, fLDA, CTM, fCTM, gpuLDA, gpuCTM.
```

## GPU Acceleration
GPU accelerating your model runs its performance bottlenecks on the GPU.

There's no reason to instantiate GPU models directly, instead you can simply instantiate the normal version of a supported model, and then use the `@gpu` macro to train it on the GPU,

```julia
model = LDA(readcorp(:nsf), 20)
@gpu train!(model, checkelbo=Inf)

## training...
```

**Important.** Notice that we did not check the ELBO at all during training. While you may check the ELBO if you wish, it's recommended that you do so infrequently, as computing the ELBO is done entirely on the CPU.

Here are the log scaled benchmarks of the coordinate ascent algorithms for the GPU models, compared against their CPU equivalents,

![GPU Benchmark](https://github.com/ericproffitt/TopicModelsVB.jl/blob/master/images/gpubar.png)

As we can see, running your model on the GPU is significantly faster than running it on the CPU.

Note that it's expected that your computer will lag when training on the GPU, since you're effectively siphoning off its rendering resources to fit your model.

## Glossary

## Types

```julia
mutable struct Document

mutable struct Corpus

abstract type TopicModel end

mutable struct LDA <: TopicModel

mutable struct fLDA <: TopicModel

mutable struct gpuLDA <: TopicModel

mutable struct CTM <: TopicModel

mutable struct fCTM <: TopicModel

mutable struct gpuCTM <: TopicModel

mutable struct CTPF <: TopicModel

mutable struct gpuCTPF <: TopicModel
```

## Corpus Functions
```julia
function check_doc

function check_corp

function readcorp

function writecorp

function abridge_corp!

function alphabetize_corp!

function remove_terms!

function compact_corp!

function condense_corp!

function pad_corp!

function remove_empty_docs!

function remove_redundant!

function stop_corp!

function trim_corp!

function trim_docs!

function fixcorp!

function showdocs

function showtitles

function getvocab

function getusers
```

## Model Functions

```julia
function showdocs

function showtitles

function check_model

function train!

@gpu train!

function gendoc

function gencorp

function showtopics

function showlibs

function showdrecs

function showurecs

function predict

function topicdist
```

## Bibliography
1. Latent Dirichlet Allocation (2003); Blei, Ng, Jordan. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)
2. Filtered Latent Dirichlet Allocation: Variational Algorithm (2016); Proffitt. [pdf](https://github.com/ericproffitt/TopicModelsVB.jl/blob/master/fLDA/fLDA.pdf)
3. Correlated Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006.pdf)
4. Content-based Recommendations with Poisson Factorization (2014); Gopalan, Charlin, Blei. [pdf](http://www.cs.columbia.edu/~blei/papers/GopalanCharlinBlei2014.pdf)
5. Numerical Optimization (2006); Nocedal, Wright. [Amazon](https://www.amazon.com/Numerical-Optimization-Operations-Financial-Engineering/dp/0387303030)
6. Machine Learning: A Probabilistic Perspective (2012); Murphy. [Amazon](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr=)
7. OpenCL in Action: How to Accelerate Graphics and Computation (2011); Scarpino. [Amazon](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173)
