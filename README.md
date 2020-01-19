# TopicModelsVB.jl

**v1.x compatible.**

A Julia package for variational Bayesian topic modeling.

Topic models are Bayesian hierarchical models designed to discover the latent low-dimensional thematic structure within corpora. Like most probabilistic graphical models, topic models are fit using either [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC), or [variational Bayesian](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VB) methods.

Markov chain Monte Carlo methods are slow but consistent, given infinite time MCMC will fit the desired model exactly. Unfortunately, the lack of an objective metric for assessing convergence means that it's difficult to state unequivocally that MCMC has reached an optimal steady-state.

Contrarily, variational Bayesian methods are fast but inconsistent, since one must approximate distributions in order to ensure tractability. Fortunately, variational Bayesian methods are fundamentally optimization algorithms, which are naturally equipped in the assessment of convergence to local optima.

This package takes the latter approach to topic modeling.

## Dependencies

```julia
DelimitedFiles
Crayons
Random
SpecialFunctions
LinearAlgebra
Distributions
OpenCL
```

## Install

```julia
Pkg.clone("https://github.com/esproff/TopicModelsVB.jl")
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

##### terms - A line of delimited positive integers corresponding to the terms which make up the document (this line is mandatory).
##### counts - A line of delimited positive integers, equal in length to terms, corresponding to the number of times a term appears in a document.
##### readers - A line of delimited positive integers corresponding to those users which have read the document.
##### ratings - A line of delimited positive integers, equal in length to readers, corresponding to the rating each reader gave the document.

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

To read a corpus into TopicModelsVB.jl, use the following function,

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
fixcorp!(corp; pad_corp=true)
```

Padding a corpus will ensure that any documents which contain vocab or user keys not in the vocab or user dictionaries are not removed. Instead, generic vocab and user keys will be added as necessary to the vocab and user dictionaries (resp.).

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
fixcorp!(corp, condense_corp=true)

showdocs(corp)
```

```
 ●●● Document 1
the fox dog quick brown jumped lazy over the
````

**Important.** A corpus is only a container for documents. 

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order vocab keys, etc.), this will not affect any corpus attached to a model. However! Since corpora are containers for their documents, modifying an individual document will affect it in all corpora which contain it. Therefore,

1. Using `fixcorp!` to modify the documents of a corpus will not result in corpus defects, but will cause them also to be changed in all other corpora which contain them.

2. Manually modifying documents is dangerous, and can result in corpus defects which cannot be fixed by `fixcorp!`. It is advised that you don't do this without good reason.

## Models
The available models are as follows:

### CPU Models
```julia
LDA(corp, K)
Latent Dirichlet Allocation model with K topics.

fLDA(corp, K)
Filtered latent Dirichlet allocation model with K topics.

CTM(corp, K)
Correlated topic model with K topics.

fCTM(corp, K)
Filtered correlated topic model with K topics.

CTPF(corp, K, basemodel)
Collaborative topic Poisson factorization model with K topics.
```

### GPU Models
```julia
gpuLDA(corp, K, batchsize)
GPU accelerated latent Dirichlet allocation model with K topics.

gpuCTM(corp, K, batchsize)
GPU accelerated correlated topic model with K topics.

gpuCTPF(corp, K, batchsize, basemodel)
GPU accelerated collaborative topic Poisson factorization model with K topics.
```

## Tutorial
### LDA
Let's begin our tutorial with a simple latent Dirichlet allocation (LDA) model with 9 topics, trained on the first 5000 documents from the NSF corpus.

```julia
using TopicModelsVB

Random.seed!(2);

corp = readcorp(:nsf) 

corp.docs = corp[1:5000];
fixcorp!(corp, trim_corp=true)
### It's strongly recommended that you trim your corpus when reducing its size in order to remove excess vocabulary. 

### Notice that the post-fix vocabulary is smaller after removing all but the first 5000 docs.

model = LDA(corp, 9)

### Setting tol=0 will ensure that all 150 iterations are completed.
### If you don't want to watch the ∆elbo, set check_elbo=Inf.
train!(model, iter=150, tol=0)

### training...

showtopics(model, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4        topic 5       topic 6      topic 7          topic 8         topic 9
data            research        species          research       research      cell         research         theory          chemistry
project         study           plant            systems        university    protein      project          problems        research
research        experimental    research         system         support       cells        data             study           metal
study           high            study            design         students      proteins     study            research        reactions
earthquake      theoretical     populations      data           program       gene         economic         equations       chemical
ocean           systems         genetic          algorithms     science       plant        important        work            study
water           phase           plants           based          scientists    studies      social           investigator    studies
studies         flow            evolutionary     control        award         genes        understanding    geometry        program
measurements    physics         population       project        dr            molecular    information      project         organic
field           quantum         data             computer       project       research     work             principal       structure
provide         materials       dr               performance    scientific    specific     development      algebraic       molecular
time            model           studies          parallel       sciences      function     theory           mathematical    dr
results         temperature     patterns         techniques     conference    system       provide          differential    compounds
models          properties      relationships    problems       national      study        analysis         groups          surface
program         dynamics        important        models         projects      important    policy           space           molecules
```

Now that we've trained our LDA model we can, if we want, take a look at the topic proportions for individual documents. For instance, document 1 has topic breakdown,

```julia
topicdist(model, 1) ### = [0.036, 0.030, 94.930, 0.036, 0.049, 0.022, 4.11, 0.027, 0.026]
```
This vector of topic weights suggests that document 1 is mostly about biology, and in fact looking at the document text confirms this observation,

```julia
showdocs(model, 1) # Could also have done showdocs(corp, 1).
```

```
 ●●● Document 1
 ●●● CRB: Genetic Diversity of Endangered Populations of Mysticete Whales: Mitochondrial DNA and Historical Demography
commercial exploitation past hundred years great extinction variation sizes
populations prior minimal population size current permit analyses effects 
differing levels species distributions life history...
```

On the other hand, some documents will be a combination of topics. Consider the topic breakdown for document 25,

```julia
model.gamma[25] # = [11.424, 45.095, 0.020, 0.036, 0.049, 0.022, 0.020, 66.573, 0.026]

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

We see that in this case document 25 appears to be about applications of mathematical physics to ocean currents, which corresponds precisely to a combination of topics 1, 2 and 8.

Furthermore, if we want to, we can also generate artificial corpora by using the ```gencorp``` function.

Generating artificial corpora will in turn run the underlying probabilistic graphical model as a generative process in order to produce entirely new collections of documents, let's try it out,

```julia
artificial_corp = gencorp(model, 5000, laplace_smooth=1e-5)
### The laplace_smooth argument governs the amount of Laplace smoothing (defaults to 0).

artificial_model = LDA(artificial_corp, 9)
train!(artificial_model, iter=150, tol=0, check_elbo=10)

### training...

showtopics(artificial_model, cols=9)
```

```
topic 1       topic 2         topic 3         topic 4          topic 5          topic 6       topic 7         topic 8          topic 9
research      research        theory          species          cell             research      data            research         chemistry
systems       study           study           research         protein          university    project         project          research
system        flow            problems        study            cells            support       research        data             reactions
design        experimental    research        plant            gene             program       study           study            metal
data          phase           equations       populations      proteins         students      earthquake      social           chemical
algorithms    high            work            genetic          genes            science       water           information      structure
based         theoretical     investigator    population       plant            award         studies         economic         studies
models        systems         project         evolutionary     studies          scientists    ocean           development      program
control       quantum         geometry        plants           molecular        project       time            work             study
problems      physics         principal       data             specific         sciences      measurements    important        molecular
computer      temperature     space           relationships    system           scientific    provide         analysis         organic
project       model           differential    understanding    research         dr            program         understanding    dr
analysis      materials       algebraic       studies          function         national      models          theory           properties
techniques    phenomena       groups          important        understanding    conference    field           models           compounds
methods       work            mathematical    patterns         study            provide       analysis        provide          reaction
```

One thing we notice so far is that despite producing what are clearly coherent topics, many of the top words in each topic are words such as *research*, *study*, *data*, etc. While such terms would be considered informative in a generic corpus, they are effectively stop words in a corpus composed of science article abstracts. Such corpus-specific stop words will be missed by most generic stop word lists, and they can be difficult to pinpoint and individually remove prior to training. Thus let's change our model to a *filtered* latent Dirichlet allocation (fLDA) model.

```julia
Random.seed!(2);

model = fLDA(corp, 9)
train!(model, iter=150, tol=0, check_elbo=Inf)

### training...

showtopics(model, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4          topic 5        topic 6       topic 7        topic 8         topic 9
earthquake      theoretical     species          design           university     cell          economic       theory          chemistry
ocean           flow            plant            algorithms       support        protein       social         equations       metal
water           phase           populations      computer         students       cells         theory         geometry        reactions
measurements    physics         genetic          parallel         program        proteins      policy         algebraic       chemical
program         quantum         plants           performance      science        gene          human          differential    program
soil            temperature     evolutionary     processing       award          plant         change         mathematical    organic
seismic         effects         population       applications     scientists     genes         political      groups          molecular
climate         phenomena       patterns         networks         scientific     molecular     public         space           compounds
effects         laser           variation        network          sciences       function      science        mathematics     surface
global          numerical       effects          software         conference     dna           decision       finite          molecules
sea             measurements    food             computational    national       regulation    people         solutions       university
surface         experiments     environmental    efficient        projects       expression    effects        spaces          reaction
response        award           ecology          program          engineering    plants        labor          dimensional     synthesis
solar           liquid          ecological       distributed      year           mechanisms    market         functions       compvocabes
earth           particle        test             power            workshop       membrane      theoretical    questions       professor
```

We can now see that many of the most troublesome corpus-specific stop words have been automatically filtered out.

### CTM
For our final example using the NSF corpus, let's upgrade our model to a filtered *correlated* topic model (fCTM).

```julia
Random.seed!(2)

model = fCTM(corp, 9)
train!(model, iter=150, tol=0, check_elbo=Inf)

### training...

showtopics(model, 20, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4          topic 5         topic 6       topic 7      topic 8         topic 9
data            flow            species          system           university      cell          data         theory          chemistry
earthquake      theoretical     plant            design           support         protein       social       problems        chemical
ocean           model           populations      data             program         cells         economic     equations       reactions
water           models          genetic          algorithms       students        plant         theory       investigator    metal
measurements    physics         evolutionary     control          science         proteins      policy       geometry        materials
program         numerical       plants           problems         dr              gene          human        algebraic       properties
climate         experimental    population       models           award           molecular     political    groups          surface
seismic         theory          data             parallel         scientists      genes         models       differential    program
soil            particle        dr               computer         scientific      dna           public       mathematical    molecular
models          dynamics        patterns         performance      sciences        system        change       space           electron
global          nonlinear       evolution        model            projects        function      model        mathematics     organic
earth           particles       variation        processing       national        regulation    science      spaces          dr
sea             quantum         group            applications     engineering     plants        people       functions       university
response        phenomena       ecology          network          conference      expression    decision     questions       molecules
damage          heat            ecological       networks         year            mechanisms    issues       manifolds       compounds
pacific         energy          forest           approach         researchers     dr            labor        finite          reaction
system          fluid           environmental    efficient        workshop        membrane      market       dimensional     laser
solar           phase           food             software         mathematical    genetic       case         properties      temperature
surface         award           experiments      computational    months          cellular      women        solutions       synthesis
ice             waves           test             distributed      equipment       binding       factors      group           optical
```

Because the topics in the fLDA model were already so well defined, there's little room for improvement in topic coherence by upgrading to the fCTM model, however what's most interesting about the CTM and fCTM models is the ability to look at topic correlations.

Based on the top 20 terms in each topic, we might tentatively assign the following topic labels:

* topic 1: *Earth Science*
* topic 2: *Physics*
* topic 3: *Ecology*
* topic 4: *Computer Science*
* topic 5: *Academia*
* topic 6: *Molecular Biology*
* topic 7: *Economics*
* topic 8: *Mathematics*
* topic 9: *Chemistry*

Now let's take a look at the topic-covariance matrix,

```julia
model.sigma

### Top 3 off-diagonal positive entries, sorted in descending order:
model.sigma[4,8] # 9.520
model.sigma[3,6] # 7.369
model.sigma[1,3] # 5.763

### Top 3 negative entries, sorted in ascending order:
model.sigma[7,9] # -14.572
model.sigma[3,8] # -12.472
model.sigma[1,8] # -11.776
```

According to the list above, the most closely related topics are topics 4 and 8, which correspond to the *Computer Science* and *Mathematics* topics, followed closely by 3 and 6, corresponding to the topics *Ecology* and *Molecular Biology*, and then by 1 and 3, corresponding to *Earth Science* and *Ecology*.

As for the most unlikely topic pairings, first are topics 7 and 9, corresponding to *Economics* and *Chemistry*, followed by topics 3 and 8, corresponding to *Ecology* and *Mathematics*, and then third are topics 1 and 8, corresponding to *Earth Science* and *Mathematics*.

Furthermore, as expected, the topic which is least correlated with all other topics is the *Academia* topic,

```julia
argmin([norm(model.sigma[:,j], 1) - model.sigma[j,j] for j in 1:9]) # = 5.
```

### Topic Prediction

The topic models so far discussed can also be used to train a classification algorithm designed to predict the topic distribution of new, unseen documents.

Let's train an LDA model on 5,000 documents, and then inspect the topic distributions for the final 10 documents in this corpus,

```julia
Random.seed!(100)

model = LDA(corp, 9)
train!(model, check_elbo=Inf)

for d in 4991:5000
	println("Document ", d, ": ", round.(topicdist(model, d), digits=3))
end
```

```
Document 4991: [0.0, 0.87, 0.128, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Document 4992: [0.0, 0.0, 0.0, 0.0, 0.173, 0.825, 0.0, 0.0, 0.0]
Document 4993: [0.171, 0.0, 0.0, 0.0, 0.0, 0.0, 0.394, 0.434, 0.0]
Document 4994: [0.0, 0.0, 0.0, 0.0, 0.92, 0.0, 0.078, 0.0, 0.0]
Document 4995: [0.002, 0.001, 0.001, 0.001, 0.656, 0.001, 0.002, 0.002, 0.334]
Document 4996: [0.001, 0.0, 0.0, 0.0, 0.001, 0.683, 0.0, 0.0, 0.314]
Document 4997: [0.001, 0.0, 0.0, 0.604, 0.001, 0.0, 0.001, 0.392, 0.0]
Document 4998: [0.177, 0.0, 0.0, 0.0, 0.001, 0.0, 0.821, 0.0, 0.0]
Document 4999: [0.406, 0.0, 0.0, 0.593, 0.0, 0.0, 0.0, 0.0, 0.0]
Document 5000: [0.001, 0.0, 0.777, 0.0, 0.22, 0.0, 0.001, 0.001, 0.0]
```

As a sanity check to make sure these topic distributions make sense, less cross-check the final two documents against the topics.

```julia
showtopics(model, cols=9)
```

```
topic 1       topic 2          topic 3      topic 4         topic 5       topic 6       topic 7       topic 8        topic 9
research      research         plant        theory          research      data          research      research       research
systems       study            cell         study           chemistry     research      dr            students       project
system        species          protein      problems        study         project       university    science        study
design        data             cells        research        chemical      earthquake    award         program        information
algorithms    project          genetic      work            reactions     time          project       university     theory
data          dr               gene         equations       studies       study         support       support        understanding
problems      important        research     investigator    high          models        program       engineering    language
methods       provide          plants       project         surface       months        scientists    conference     important
models        studies          species      geometry        metal         support       physics       provide        behavior
based         analysis         studies      principal       materials     year          study         projects       social
control       patterns         genes        mathematical    organic       ocean         professor     project        data
analysis      history          molecular    groups          properties    important     field         national       work
parallel      evolutionary     proteins     algebraic       structure     analysis      laboratory    sciences       economic
project       information      dna          differential    molecular     model         studies       faculty        political
techniques    relationships    study        space           program       seismic       institute     scientific     models
```

```julia
showdocs(model, 4999:5000)
```

```
 ●●● Document 4999
 ●●● Uses and Simulation of Randomness: Applications to Cryptography,Program Checking and Counting Problems.
randomized algorithms consume valuable resource uniformly distributed random bits primary focuses work develop general techniques designing pseudo generator stretches short string longer totally polynomial time adversary central component secure private key cryptosystem conserve number monte carlo shown construct function investigator plans constructions efficient generators typical important counting problem estimate truth assignments satisfy boolean formula algorithm designed deterministic sought recently theory program checking developed supplement verification testing computing verifying correctness answer possibly partially faulty supposedly computes successfully applied variety algebraic problems extended applications

 ●●● Document 5000
 ●●● New Possibilities for Understanding the Role of Neuromelanin
class cellular found tissues organisms product metabolism pigment central nervous system mammals including man involved production dopamine function unknown indirect evidence important determining behavior neurons genesis goal project determine role studying vivo scanning electron microscopy magnetic resonance imaging techniques monitor paramagnetic semiconductor properties relate functional states investigation increase understanding treatment disease disorders
```

Now as a performance test, we can partition our corpus into training and test corpora as follows,

```julia
train_corp = copy(corp)
train_corp.docs = train_corp[1:4990];

test_corp = copy(corp)
test_corp.docs = test_corp[4991:5000];
```

Finally, we will retrain our LDA model on just the training corpus, and then used that trained model to predict the topic distributions of the ten documents in our test corpus,

```julia
Random.seed!(100);

train_model = LDA(train_corp, 9)
train!(model, check_elbo=Inf)

test_model = predict(test_corp, train_model=train_model)
```

The `predict` function works by taking in a corpus of new, unseen documents, and a trained model, and returning a new model of the same type. This new model can then be inspected directly, or using `topicdist`, in order to see the topic distribution for the documents in the test corpus.

```julia
for d in 4991:5000
	println("Document ", 4990 + d, ": ", round.(topicdist(model, d), digits=3))
end
```

```
Document 4991: [0.035, 0.51, 0.31, 0.003, 0.035, 0.089, 0.011, 0.003, 0.004]
Document 4992: [0.002, 0.046, 0.175, 0.001, 0.13, 0.604, 0.004, 0.037, 0.001]
Document 4993: [0.086, 0.0, 0.0, 0.028, 0.184, 0.006, 0.218, 0.478, 0.0]
Document 4994: [0.0, 0.0, 0.0, 0.0, 0.898, 0.0, 0.099, 0.0, 0.0]
Document 4995: [0.182, 0.001, 0.001, 0.001, 0.515, 0.001, 0.002, 0.002, 0.294]
Document 4996: [0.195, 0.042, 0.0, 0.0, 0.001, 0.598, 0.0, 0.0, 0.163]
Document 4997: [0.001, 0.0, 0.0, 0.535, 0.026, 0.0, 0.001, 0.405, 0.031]
Document 4998: [0.129, 0.066, 0.0, 0.0, 0.08, 0.152, 0.536, 0.0, 0.036]
Document 4999: [0.405, 0.002, 0.016, 0.489, 0.002, 0.002, 0.002, 0.057, 0.023]
Document 5000: [0.001, 0.03, 0.76, 0.0, 0.207, 0.0, 0.001, 0.001, 0.0]
```

### CTPF
For our final model, we take a look at the collaborative topic Poisson factorization (CTPF) model. CTPF is a collaborative filtering topic model which uses the latent thematic structure of documents to improve the quality of document recommendations beyond what would be possible using just the document-user matrix alone. This blending of thematic structure with known user prefrences not only improves recommendation accuracy, but also mitigates the cold-start problem of recommending to users never-before-seen documents. As an example, let's load the CiteULike dataset into a corpus and then randomly remove a single reader from each of the documents.

```julia
Random.seed!(2);

corp = readcorp(:citeu)

ukeys_test = Int[]
for doc in corp
    index = sample(1:length(doc.readers), 1)[1]
    push!(ukeys_test, doc.readers[index])
    deleteat!(doc.readers, index)
    deleteat!(doc.ratings, index)
end
```

**Important.** We refrain from fixing our corpus in this case, first because the CiteULike dataset is pre-packaged and thus pre-fixed, but more importantly, because removing user keys from documents and then fixing a corpus may result in a re-ordering of its user dictionary, which would in turn invalidate our test set.

After training, we will evaluate model quality by measuring our model's success at imputing the correct user back into each of the document libraries.

It's also worth noting that after removing a single reader from each document, 158 of the documents now have 0 readers,

```julia
sum([isempty(doc.readers) for doc in corp]) # = 158
```

Fortunately, since CTPF can if need be depend entirely on thematic structure when making recommendations, this poses no problem for the model.

Now that we have set up our experiment, we instantiate and train a CTPF model on our corpus. Furthermore, since we're not interested in the interpretability of the topics, we'll instantiate our model with a larger than usual number of topics (K=30), and then run it for a relatively short number of iterations (iter=20).

```julia
Random.seed!(2);

model = gpuCTPF(corp, 30)
train!(model, iter=20, check_elbo=5)

### training...
```

Finally, we evaluate the accuracy of our model against the test set, where baseline for mean accuracy is 0.5.

```julia
accuracy = Float64[]
for (d, u) in enumerate(ukeys_test)
    urank = findall(model.drecs[d] .== u)[1]
    nrlen = length(model.drecs[d])
    push!(accuracy, (nrlen - urank) / (nrlen - 1))
end

@show mean(accuracy) # mean(accuracy) = 0.903
```

We can see that, on average, our model ranks the true hidden reader in the top 9.7% of all non-readers for each document.

Let's also take a look at the top recommendations for a particular document,

```julia
ukeys_test[1] # = 3741
accuracy[1] # = 0.965
### user3741's library test document was placed in the top 3.5% of his or her recommendations.

showdrecs(model, 1, 192)
```
```
 ●●● doc 1
 ●●● The metabolic world of Escherichia coli is not small
...
188. #user4803
189. #user3652
190. #user1254
191. #user5052
192. #user3741
```

For evaluating user recommendations, let's take a more holistic approach.

Since large heterogenous libraries make the qualitative assessment of recommendations difficult, let's search for a user with a modestly sized relatively focused library,

```julia
showlibs(model, 1741)
```

```
 ●●● user 1741
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
 
 Now compare this with the top 50 recommendations (the top 0.3%) made by our model,
 
```julia
showurecs(model, 1741, 50)
```

```
 ●●● user 1741
1.  On Understanding Types, Data Abstraction, and Polymorphism
2.  Can programming be liberated from the von {N}eumann style? {A} functional style and its algebra of programs
3.  Haskell's overlooked object system
4.  Contracts for higher-order functions
5.  Why Dependent Types Matter
6.  Modern {C}ompiler {I}mplementation in {J}ava
7.  Generalising monads to arrows
8.  Functional programming with bananas, lenses, envelopes and barbed wire
9.  Featherweight Java: A Minimal Core Calculus for Java and GJ
10. On the expressive power of programming languages
11. Ownership types for safe programming: preventing data races and deadlocks
12. Dependent Types in Practical Programming
13. The essence of compiling with continuations
14. Types and programming languages
15. A {S}yntactic {T}heory of {D}ynamic {B}inding
16. Principles of programming with compvocab objects and collection types
17. Functional pearl: implicit configurations--or, type classes reflect the values of types
18. Typed Memory Management in a Calculus of Capabilities
19. Type Classes with Functional Dependencies
20. Macros as multi-stage computations: type-safe, generative, binding macros in MacroML
21. Dynamic optimization for functional reactive programming using generalized algebraic data types
22. Monadic Parsing in Haskell
23. Monadic Parser Combinators
24. Types, abstraction and parametric polymorphism
25. Fast and Loose Reasoning Is Morally Correct
26. Scrap your boilerplate: a practical design pattern for generic programming
27. A comparative study of language support for generic programming
28. Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I
29. Definitional interpreters for higher-order programming languages
30. A new notation for arrows
31. The design and implementation of typed scheme
32. A theory of type polymorphism in programming
33. Sets for Mathematics
34. Adoption and focus: practical linear types for imperative programming
35. Languages of the Future
36. The {Calculus of Constructions}
37. Scrap your nameplate: (functional pearl)
38. Foundations for structured programming with GADTs
39. Synthesizing Object-Oriented and Functional Design to Promote Re-Use
40. A Tutorial on (Co)Algebras and (Co)Induction
41. The next 700 programming languages
42. Visual Programming
43. The Definition of Standard ML - Revised
44. Call-by-name, call-by-value, and the $\lambda$-calculus
45. Stack-Based Typed Assembly Language
46. Associated types with class
47. Programming Languages: Application and Interpretation
48. Type Systems
49. Ott: Effective Tool Support for the Working Semanticist
50. Typed Contracts for Functional Programming
```

Note, as was done by Blei et al. in their original paper, if you would like to warm start your CTPF model using the topic distributions generated by one of the other models, simply do the following prior to training your model,

```julia
ctpf_model.alef = exp.(model.beta)
### For model of type: LDA, fLDA, CTM, fCTM, gpuLDA, gpuCTM.
```   

## GPU Acceleration
GPU accelerating your model runs its performance bottlenecks on the GPU rather than the CPU.

There's no reason to instantiate the GPU models directly, instead you can simply instantiate the normal version of a supported model, and then use the `@gpu` macro to train it on the GPU,

```julia
corp = readcorp(:nsf)

model = LDA(corp, 16)
@time @gpu train!(model, iter=150, tol=0, chkelbo=151)
### Let's time it as well to get an exact benchmark. 

### training...

### 120.928241 seconds (130.06 M allocations: 22.745 GiB, 3.60% gc time)
### On an Intel Iris Plus Graphics 640 1536 MB GPU.
```

This algorithm just crunched through a 16 topic 128,804 document topic model in 2 minutes.

**Important.** Notice that we didn't check the ELBO at all during training. While you can check the ELBO if you wish, it's recommended that you do so infrequently, since checking the ELBO for GPU models requires expensive memory allocations on the CPU.

Here is the benchmark of the above model's coordinate ascent algorithm against the equivalent algorithm run on the CPU,

![GPU Benchmark](https://github.com/ericproffitt/TopicModelsVB.jl/blob/version-1.x/images/ldabar.png)

As we can see, running the LDA model on the GPU is approximatey 3200% faster than running it on the CPU.

Note, it's expected that your computer will lag when training on the GPU, since you're effectively siphoning off its rendering resources to fit your model.

## Types

```julia
mutable struct Document
	"Document mutable struct"

	"terms:   A vector{Int} containing keys for the Corpus vocab Dict."
	"counts:  A Vector{Int} denoting the counts of each term in the Document."
	"readers: A Vector{Int} denoting the keys for the Corpus users Dict."
	"ratings: A Vector{Int} denoting the ratings for each reader in the Document."
	"title:   The title of the document (String)."

	terms::Vector{Int}
	counts::Vector{Int}
	readers::Vector{Int}
	ratings::Vector{Int}
	title::String

mutable struct Corpus
	"Corpus mutable struct."

	"docs:  A Vector{Document} containing the documents which belong to the Corpus."
	"vocab: A Dict{Int, String} containing a mapping term Int (key) => term String (value)."
	"users: A Dict{Int, String} containing a mapping user Int (key) => user String (value)."

	docs::Vector{Document}
	vocab::Dict{Int, String}
	users::Dict{Int, String}

abstract type TopicModel end

mutable struct LDA <: TopicModel
	"LDA mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct fLDA <: TopicModel
	"fLDA mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct CTM <: TopicModel
	"CTM mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct fCTM <: TopicModel
	"fCTM mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct CTPF <: TopicModel
	"CTPF mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct gpuLDA <: TopicModel
	"gpuLDA mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct gpuCTM <: TopicModel
	"gpuCTM mutable struct."

	corpus::Corpus
	K::Int
	...

mutable struct gpuCTPF <: TopicModel
	"gpuCTPF mutable struct."

	corpus::Corpus
	K::Int
	...
```

### Document/Corpus Functions
```julia
function check_doc(doc::Document)
	"Check Document parameters."

function check_corp(corp::Corpus)
	"Check Corpus parameters."

function readcorp(;docfile::String="", vocabfile::String="", userfile::String="", titlefile::String="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	"Load a Corpus object from text file(s)."

	### readcorp(:nsf)   	- National Science Foundation Corpus.
	### readcorp(:citeu)	- CiteULike Corpus.

function writecorp(corp::Corpus; docfile::String="", vocabfile::String="", userfile::String="", titlefile::String="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false)	
	"Write a corpus."

function abridge_corp!(corp::Corpus, n::Integer=0)
	"All terms which appear less than or equal to n times in the corpus are removed from all documents."

function alphabetize_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Alphabetize vocab and/or user dictionaries."

function compact_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Relabel vocab and/or user keys so that they form a unit range."

function condense_corp!(corp::Corpus)
	"Ignore term order in documents."
	"Multiple seperate occurrences of terms are stacked and their associated counts increased."

function pad_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Enter generic values for vocab and/or user keys which appear in documents but not in the vocab/user dictionaries."

function remove_empty_docs!(corp::Corpus)
	"Documents with no terms are removed from the corpus."

function remove_redundant!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Remove vocab and/or user keys which map to redundant values."
	"Reassign Document term and/or reader keys."

function stop_corp!(corp::Corpus)
	"Filter stop words in the associated corpus."

function trim_corp!(corp::Corpus; vocab::Bool=true, users::Bool=true)
	"Those keys which appear in the corpus vocab and/or user dictionaries but not in any of the documents are removed from the corpus."

function trim_docs!(corp::Corpus; terms::Bool=true, readers::Bool=true)
	"Those vocab and/or user keys which appear in documents but not in the corpus dictionaries are removed from the documents."

function fixcorp!(corp::Corpus; vocab::Bool=true, users::Bool=true, abridge_corp::Integer=0, alphabetize_corp::Bool=false, compact_corp::Bool=false, condense_corp::Bool=false, pad_corp::Bool=false, remove_empty_docs::Bool=false, remove_redundant::Bool=false, stop_corp::Bool=false, trim_corp::Bool=false)
	"Generic function to ensure that a Corpus object can be loaded ino a TopicModel object."
	"Contains optional keyword arguments."

function showdocs(corp::Corpus, doc_indices)
	"Display document(s) in readable format."

function getvocab(corp::Corpus)

function getusers(corp::Corpus)
```

### Model Functions

```julia
function showdocs(model::TopicModel, docs)
	"Display document(s) in readable format."

function check_model(model::TopicModel)
	"Check model parameters."

function train!(model::TopicModel; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	"Train TopicModel."

	### 'iter'		- maximum number of iterations through the corpus.
	### 'tol'		- absolute tolerance for ∆elbo as a stopping criterion.
	### 'niter'		- maximum number of iterations for Newton's and interior-point Newton's methods. (not included for CTPF and gpuCTPF models.)
	### 'ntol'		- tolerance for change in function value as a stopping criterion for Newton's and interior-point Newton's methods. (not included for CTPF and gpuCTPF models.)
	### 'viter'		- maximum number of iterations for optimizing variational parameters (at the document level).
	### 'vtol'		- tolerance for change in variational parameter values as stopping criterion.
	### 'check_elbo'	- number of iterations between ∆elbo checks (for both evaluation and convergence of the evidence lower-bound).

@gpu train!
	"Train model on GPU."

function gendoc(model::TopicModel, laplace_smooth::Real=0.0)
	"Generate a generic document from model parameters by running the associated graphical model as a generative process."

function gencorp(model::TopicModel, corpsize::Int, laplace_smooth::Real=0.0)
	"Generate a generic corpus of size 'corpsize' from model parameters."

function showtopics(model::TopicModel, N::Integer=min(15, model.V); topics::Union{Integer, Vector{Integer}}=collect(1:model.K), cols::Integer=4)
	"Display the top 'N' words for each topic in 'topics', defaults to 4 columns per line."

function showlibs(model::Union{CTPF, gpuCTPF}, users)
	"Show the document(s) in a user's library."

function showdrecs(model::Union{CTPF, gpuCTPF}, docs, U)
	"Show the top 'U' user recommendations for a document(s)."

function showurecs(ctpf::Union{CTPF, gpuCTPF}, users::Union{Integer, Vector{Integer}}, M::Integer=min(10, ctpf.M))
	"Show the top 'M' document recommendations for a user(s)."
```

## Bibliography
1. Latent Dirichlet Allocation (2003); Blei, Ng, Jordan. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)
2. Filtered Latent Dirichlet Allocation: Variational Bayes Algorithm (2016); Proffitt. [pdf](https://github.com/esproff/TopicModelsVB.jl/blob/master/fLDAVB.pdf)
3. Correlated Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006.pdf)
4. Content-based Recommendations with Poisson Factorization (2014); Gopalan, Charlin, Blei. [pdf](http://www.cs.columbia.edu/~blei/papers/GopalanCharlinBlei2014.pdf)
5. Numerical Optimization (2006); Nocedal, Wright. [Amazon](https://www.amazon.com/Numerical-Optimization-Operations-Financial-Engineering/dp/0387303030)
6. Machine Learning: A Probabilistic Perspective (2012); Murphy. [Amazon](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr=)
7. OpenCL in Action: How to Accelerate Graphics and Computation (2011); Scarpino. [Amazon](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173)
