# TopicModelsVB.jl

**v0.6 compatible**

A Julia Package for Variational Bayesian Topic Modeling.

Topic models are Bayesian hierarchical models designed to discover the latent low-dimensional thematic structure within corpora.  Topic models, like most probabilistic graphical models, are fit using either [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC), or [variational Bayesian](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VB) methods.

Markov chain Monte Carlo methods are slower but consistent, given infinite time MCMC will fit the desired model exactly.  Unfortunately, the lack of an objective metric for assessing convergence means that within any finite time horizon it's difficult to state unequivocally that MCMC has reached an optimal steady-state.

Contrarily, variational Bayesian methods are faster but inconsistent, since one must approximate distributions in order to ensure tractability.  Furthermore, variational Bayesian methods, being numerical optimization procedures, are naturally equipped for the assessment of convergence to local optima.  This package takes the latter approach to topic modeling.

**Important:** If you find a bug, please don't hesitate to open an issue, I should reply promptly.

## Dependencies

```julia
SpecialFunctions
Distributions
OpenCL
```

## Install

```julia
Pkg.clone("https://github.com/esproff/TopicModelsVB.jl")
```

## Datasets
Included in TopicModelsVB.jl are three datasets:

1. National Science Foundation Abstracts 1989 - 2003:
  * 128804 documents
  * 25319 lexicon

2. CiteULike Science Article Database:
  * 16980 documents
  * 8000 lexicon
  * 5551 users

3. Macintosh Magazine Article Collection 1984 - 2005:
  * 75011 documents
  * 15113 lexicon

## Corpus
Let's begin with the Corpus data structure.  The Corpus data structure has been designed for maximum ease-of-use.  Datasets must still be cleaned and put into the appropriate format, but once a dataset is in the proper format and read into a corpus, it can easily be modified to meet the user's needs.

There are four plaintext files that make up a corpus:
 * docfile
 * lexfile
 * userfile
 * titlefile
 
None of these files are mandatory to read a corpus, and in fact reading no files will result in an empty corpus.  However in order to train a model a docfile will be necessary, since it contains all quantitative data known about the documents.  On the other hand, the lex, user and title files are used solely for interpreting output.

The docfile should be a plaintext file containing lines of delimited numerical values.  Each document is a block of lines, the number of which depends on what information is known about the documents.  Since a document is at its essence a list of terms, each document *must* contain at least one line containing a nonempty list of delimited positive integer values corresponding to the terms of which it is composed.  Any further lines in a document block are optional, however if they are present they must be present for all documents and must come in the following order:

##### hello

<dl>
 <dt>terms</dt>
 <dd>A line of delimited positive integers corresponding to the terms which make up the document (this line is mandatory).</dd>
 <dt>counts</dt>
 <dd><font size="6">A line of delimited positive integers, equal in length to the term line, corresponding to the number of times a particular term appears in a document.</font></dd>
 <dt>readers</dt>
 <dd>A line of delimited positive integers corresponding to those users which have read the document.</dd>
 <dt>ratings</dt>
 <dd>A line of delimited positive integers, equal in length to the readers line, corresponding to the rating each reader gave the document.</dd>
 <dt>stamp</dt>
 <dd>A numerical value in the range [-Inf, Inf] denoting the timestamp of the document.</dd>
</dl>

An example of a single doc block from a docfile with all possible lines included:

```
...
4,10,3,100,57
1,1,2,1,3
1,9,10
1,1,5
19990112.0
...
```

The lex and user files are tab delimited dictionaries mapping positive integers to terms and usernames (resp.).  For example,

```
1    this
2    is
3    a
4    lex
5    file
```

A userfile is identitcal to a lexfile, except usernames will appear in place of vocabulary terms.

Finally, a titlefile is simply a list of titles, not a dictionary, and is of the form:

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
readcorp(;docfile="", lexfile="", userfile="", titlefile="", delim=',', counts=false, readers=false, ratings=false, stamps=false)
```

The ```file``` keyword arguments indicate the path where the respective file is located.

It is often the case that even once files are correctly formatted and read, the corpus will still contain formatting defects which prevent it from being loaded into a model.  Therefore, before loading a corpus into a model, it is **very important** that one of the following is run:

```julia
fixcorp!(corp; kwargs...)
```

or

```julia
padcorp!(corp; kwargs...)
fixcorp!(corp; kwargs...)
```

Padding a corpus before fixing it will ensure that any documents which contain lex or user keys not in the lex or user dictionaries are not removed.  Instead, generic lex and user keys will be added as necessary to the lex and user dictionaries (resp.).

**Important:** A corpus is only a container for documents.  

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order lex keys, etc.), this will not affect any corpus attached to a model.  However!  Since corpora are containers for their documents, modifying an individual document will affect this document in all corpora which contain it.  Therefore:

**1. Using `corp!` functions to modify the documents of a corpus will not result in corpus defects, but will cause them also to be changed in all other corpora which contain them.**

**2. Manually modifying documents is dangerous, and can result in corpus defects which cannot be fixed by `fixcorp!`.  It's advised that you don't do this with out a good reason.**

## Models
The available models are as follows:

### Models
```julia
LDA(corp, K)
# Latent Dirichlet Allocation model with K topics.

fLDA(corp, K)
# Filtered latent Dirichlet allocation model with K topics.

CTM(corp, K)
# Correlated topic model with K topics.

fCTM(corp, K)
# Filtered correlated topic model with K topics.

DTM(corp, K, delta, basemodel)
# Dynamic topic model with K topics and ∆ = delta.

CTPF(corp, K, basemodel)
# Collaborative topic Poisson factorization model with K topics.
```

### GPU Accelerated Models
```julia
gpuLDA(corp, K, batchsize)
# GPU accelerated latent Dirichlet allocation model with K topics.

gpuCTM(corp, K, batchsize)
# GPU accelerated correlated topic model with K topics.

gpuCTPF(corp, K, batchsize, basemodel)
# GPU accelerated collaborative topic Poisson factorization model with K topics.
```

## Tutorial
### LDA
Let's begin our tutorial with a simple latent Dirichlet allocation (LDA) model with 9 topics, trained on the first 5000 documents from the NSF corpus.

```julia
using TopicModelsVB

srand(2)

corp = readcorp(:nsf) 

corp.docs = corp[1:5000]
fixcorp!(corp)

# Notice that the post-fix lexicon is smaller after removing all but the first 5000 docs.

model = LDA(corp, 9)
train!(model, iter=150, tol=0) # Setting tol=0 will ensure that all 150 iterations are completed.
                               # If you don't want to watch the ∆elbo, set chkelbo=151.
# training...

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

Now that we've trained our LDA model we can, if we want, take a look at the topic proportions for individual documents.  For instance, document 1 has topic breakdown:

```julia
model.gamma[1] # = [0.036, 0.030, 94.930, 0.036, 0.049, 0.022, 4.11, 0.027, 0.026]
```
This vector of topic weights suggests that document 1 is mostly about biology, and in fact looking at the document text confirms this observation:

```julia
showdocs(model, 1) # Could also have done showdocs(corp, 1).
```

```
 ●●● doc 1
 ●●● CRB: Genetic Diversity of Endangered Populations of Mysticete Whales: Mitochondrial DNA and Historical Demography
commercial exploitation past hundred years great extinction variation sizes
populations prior minimal population size current permit analyses effects 
differing levels species distributions life history...
```

On the other hand, some documents will be a combination of topics.  Consider the topic breakdown for document 25:

```julia
model.gamma[25] # = [11.424, 45.095, 0.020, 0.036, 0.049, 0.022, 0.020, 66.573, 0.026]

showdocs(model, 25)
```

```
 ●●● doc 25
 ●●● Mathematical Sciences: Nonlinear Partial Differential Equations from Hydrodynamics
work project continues mathematical research nonlinear elliptic problems arising perfect
fluid hydrodynamics emphasis analytical study propagation waves stratified media techniques
analysis partial differential equations form basis studies primary goals understand nature 
internal presence vortex rings arise density stratification due salinity temperature...
```

We see that in this case document 25 appears to be about applications of mathematical physics to ocean currents, which corresponds precisely to a combination of topics 1, 2 and 8.

Furthermore, if we want to, we can also generate artificial corpora by using the ```gencorp``` function.  Generating artificial corpora will in turn run the underlying probabilistic graphical model as a generative process in order to produce entirely new collections of documents, let's try it out:

```julia
artifcorp = gencorp(model, 5000, 1e-5) # The third argument governs the amount of Laplace smoothing (defaults to 0).

artifmodel = LDA(artifcorp, 9)
train!(artifmodel, iter=150, tol=0, chkelbo=15)

# training...

showtopics(artifmodel, cols=9)
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

One thing we notice so far is that despite producing what are clearly coherent topics, many of the top words in each topic are words such as *research*, *study*, *data*, etc.  While such terms would be considered informative in a generic corpus, they are effectively stop words in a corpus composed of science article abstracts.  Such corpus-specific stop words will be missed by most generic stop word lists, and they can be difficult to pinpoint and individually remove prior to training.  Thus let's change our model to a *filtered* latent Dirichlet allocation (fLDA) model.

```julia
srand(2)

model = fLDA(corp, 9)
train!(model, iter=150, tol=0)

# training...

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
solar           liquid          ecological       distributed      year           mechanisms    market         functions       complexes
earth           particle        test             power            workshop       membrane      theoretical    questions       professor
```

We can now see that many of the most troublesome corpus-specific stop words have been automatically filtered out, while those that remain are those which tend to cluster within their own, more generic, topic.

### CTM
For our final example using the NSF corpus, let's upgrade our model to a filtered *correlated* topic model (fCTM).

```julia
srand(2)

model = fCTM(corp, 9)
train!(model, iter=150, tol=0)

# training...

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

Now let's take a look at the topic-covariance matrix:

```julia
model.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
model.sigma[4,8] # 9.520
model.sigma[3,6] # 7.369
model.sigma[1,3] # 5.763

# Top 3 negative entries, sorted in ascending order:
model.sigma[7,9] # -14.572
model.sigma[3,8] # -12.472
model.sigma[1,8] # -11.776
```

According to the list above, the most closely related topics are topics 4 and 8, which correspond to the *Computer Science* and *Mathematics* topics, followed closely by 3 and 6, corresponding to the topics *Ecology* and *Molecular Biology*, and then by 1 and 3, corresponding to *Earth Science* and *Ecology*.

As for the most unlikely topic pairings, first are topics 7 and 9, corresponding to *Economics* and *Chemistry*, followed by topics 3 and 8, corresponding to *Ecology* and *Mathematics*, and then third are topics 1 and 8, corresponding to *Earth Science* and *Mathematics*.

Furthermore, as expected, the topic which is least correlated with all other topics is the *Academia* topic:

```julia
indmin([norm(model.sigma[:,j], 1) - model.sigma[j,j] for j in 1:9]) # = 5.
```

### DTM
Now that we have covered static topic models, let's transition to the dynamic topic model (DTM).  The dynamic topic model discovers the temporal-dynamics of topics which, nevertheless, remain thematically static.  A good example of a topic which is thematically-static, yet exhibits an evolving lexicon, is *Computer Storage*.  Methods of data storage have evolved rapidly in the last 40 years, evolving from punch cards, to 5-inch floppy disks, to smaller hard disks, to zip drives and cds, to dvds and platter hard drives, and now to flash drives, solid-state drives and cloud storage, all accompanied by the rise and fall of computer companies which manufacture (or at one time manufactured) these products.

As an example, let's load the Macintosh corpus of articles, drawn from the magazines *MacWorld* and *MacAddict*, published between the years 1984 - 2005.  We sample 400 articles randomly from each year, and break time periods into 2 year intervals.

```julia
import Distributions.sample

srand(1)

corp = readcorp(:mac)

corp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, corp.docs), 400, replace=false) for y in 1984:2005]...)

fixcorp!(corp, abr=100, len=10) # Remove words which appear < 100 times and documents of length < 10.

basemodel = LDA(corp, 9)
train!(basemodel, iter=150, chkelbo=151)

# training...

model = DTM(corp, 9, 200, basemodel)
train!(model, iter=10) # This will likely take about an hour on a personal computer.
                       # Convergence for all other models is worst-case quadratic,
                       # while DTM convergence is linear or at best super-linear.
# training...
```

We can look at a particular topic slice, in this case the *Macintosh Hardware* topic, by writing:

```julia
showtopics(model, topics=8, cols=11)
```

```
 ●●● Topic: 8
time 1       time 2       time 3       time 4        time 5         time 6         time 7        time 8         time 9       time 10      time 11
disk         macintosh    color        drive         mac            power          drive         drive          usb          mac          ipod
memory       disk         drive        mac           drive          drive          ram           scsi           drive        usb          usb
hard         drive        disk         drives        powerbook      quadra         memory        g3             ram          firewire     power
ram          memory       scsi         hard_drive    color          apple          power         ram            firewire     g4           mini
disks        scsi         drives       simms         board          scsi           hard_drive    drives         scsi         drive        memory
macintosh    hard         hard         board         drives         drives         processor     power          g4           power        drive
port         drives       board        meg           scsi           ram            disk          usb            power        ram          ram
board        ncp          macintosh    removable     ram            video          speed         speed          g3           apple        airport
power        color        ram          system        power          powerbook      faster        powerbook      memory       memory       firewire
drives       ram          meg          power         quadra         performance    pci           apple          port         port         performance
external     port         software     external      memory         data           monitor       memory         hardware     imac         g5
serial       internal     memory       memory        system         speed          drives        external       faster       drives       mac
software     software     speed        ram           speed          upgrade        video         performance    processor    powerbook    faster
speed        external     system       storage       port           modem          macs          processor      drives       storage      models
internal     power        external     data          accelerator    duo            slots         serial         speed        dual         apple
```

or a particular time slice, by writing:

```julia
showtopics(model, times=11, cols=9)
```

```
 ●●● Time: 11
 ●●● Span: 200405.0 - 200512.0
topic 1    topic 2      topic 3     topic 4       topic 5        topic 6      topic 7     topic 8       topic 9
file       color        system      ipod          demo           click        apple       drive         mac
select     image        files       mini          manager        video        site        express       mouse
folder     images       disk        power         future         software     price       backup        cover
set        photo        osx         usb           director       music        web         drives        fax
open       photoshop    utility     apple         usa            audio        products    buffer        software
menu       print        install     g5            network        good         contest     prices        ea
choose     light        finder      firewire      shareware      game         smart       subject       pad
text       photos       user        g4            charts         itunes       year        data          laserwriter
button     printer      terminal    ram           accounts       play         computer    warranty      kensington
find       mode         run         models        editor         time         world       mac           stylewriter
window     digital      folders     display       advertising    effects      phone       disk          turbo
type       quality      network     memory        entries        pro          group       retrospect    apple
create     elements     classic     hard_drive    production     dvd          product     orders        printer
press      lens         desktop     speed         california     makes        service     notice        modem
line       printing     windows     port          marketing      interface    people      shipping      ext
```

As you may have noticed, the dynamic topic model is *extremely* computationally intensive, it is likely that running the DTM model on an industry-sized dataset will always require more computational power than can be provided by your standard personal computer.

### CTPF
For our final model, we take a look at the collaborative topic Poisson factorization (CTPF) model.  CTPF is a collaborative filtering topic model which uses the latent thematic structure of documents to improve the quality of document recommendations beyond what would be possible using just the document-user matrix alone.  This blending of thematic structure with known user prefrences not only improves recommendation accuracy, but also mitigates the cold-start problem of recommending to users never-before-seen documents.  As an example, let's load the CiteULike dataset into a corpus and then randomly remove a single reader from each of the documents.

```julia
import Distributions.sample

srand(1)

corp = readcorp(:citeu)

testukeys = Int[]
for doc in corp
    index = sample(1:length(doc.readers), 1)[1]
    push!(testukeys, doc.readers[index])
    deleteat!(doc.readers, index)
    deleteat!(doc.ratings, index)
end
```

**Important:** We refrain from fixing our corpus in this case, first because the CiteULike dataset is pre-packaged and thus pre-fixed, but more importantly, because removing user keys from documents and then fixing a corpus may result in a re-ordering of its user dictionary, which would in turn invalidate our test set.

After training, we will evaluate model quality by measuring our model's success at imputing the correct user back into each of the document libraries.

It's also worth noting that after removing a single reader from each document, 158 of the documents now have 0 readers:

```julia
sum([isempty(doc.readers) for doc in corp]) # = 158
```

Fortunately, since CTPF can if need be depend entirely on thematic structure when making recommendations, this poses no problem for the model.

Now that we have set up our experiment, we instantiate and train a CTPF model on our corpus.  Furthermore, since we're not interested in the interpretability of the topics, we'll instantiate our model with a larger than usual number of topics (K=30), and then run it for a relatively short number of iterations (iter=20).

```julia
srand(1)

model = CTPF(corp, 30) # Note: If no 'basemodel' is entered then parameters will be initialized at random.
train!(model, iter=20)

# training...
```

Finally, we evaluate the accuracy of our model against the test set, where baseline for mean accuracy is 0.5.

```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(model.drecs[d], u)[1]
    nrlen = length(model.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.910
```

Not bad, but let's see if we can't improve our accuracy at least a bit by priming our CTPF model with a 100 iteration LDA model.

In the interest of time, let's use the GPU accelerated verions of LDA and CTPF:


```julia
srand(1)

basemodel = gpuLDA(corp, 30)
train!(basemodel, iter=100, chkelbo=101)

# training...

model = gpuCTPF(corp, 30, basemodel)
train!(model, iter=20, chkelbo=21)

# training...
```

Again we evaluate the accuracy of our model against the test set:

```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(model.drecs[d], u)[1]
    nrlen = length(model.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.916
```

We can see that, on average, our model ranks the true hidden reader in the top 8.4% of all non-readers for each document.

Let's also take a look at the top recommendations for a particular document(s):

```julia
testukeys[1] # = 997
acc[1] # = 0.981
# user997's library test document was placed in the top 2% of his or her recommendations.

showdrecs(model, 1, 106, cols=1)
```
```
 ●●● doc 1
 ●●● The metabolic world of Escherichia coli is not small
...
102. #user1658
103. #user2725
104. #user1481
105. #user5380
106. #user997
```
as well as those for a particular user(s):

```julia
showurecs(model, 997, 578)
```
```
 ●●● user 997
...
574.  Life-history trade-offs favour the evolution of animal personalities
575.  Coupling and coordination in gene expression processes: a systems biology view.
576.  Principal components analysis to summarize microarray experiments: application to sporulation time series
577.  Rich Probabilistic Models for Gene Expression
578.  The metabolic world of Escherichia coli is not small
```

We can also take a more holistic approach to evaluating model quality.

Since large heterogenous libraries make the qualitative assessment of recommendations difficult, let's search for a user with a modestly sized relatively focused library:

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
 
 The 20 articles in user 1741's library suggest that his or her interests lie at the intersection of programming language theory and foundational mathematics.  
 
 Now compare this with the top 20 recommendations made by our model:
 
```julia
showurecs(model, 1741, 20)
```

```
 ●●● user 1741
1.  Sets for Mathematics
2.  On Understanding Types, Data Abstraction, and Polymorphism
3.  Can programming be liberated from the von {N}eumann style? {A} functional style and its algebra of programs
4.  Haskell's overlooked object system
5.  Contracts for higher-order functions
6.  Principles of programming with complex objects and collection types
7.  Ownership types for safe programming: preventing data races and deadlocks
8.  Modern {C}ompiler {I}mplementation in {J}ava
9.  Functional pearl: implicit configurations--or, type classes reflect the values of types
10. Featherweight Java: A Minimal Core Calculus for Java and GJ
11. On the expressive power of programming languages
12. Typed Memory Management in a Calculus of Capabilities
13. Dependent Types in Practical Programming
14. Functional programming with bananas, lenses, envelopes and barbed wire
15. The essence of compiling with continuations
16. Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I
17. Visual Programming
18. Dynamic optimization for functional reactive programming using generalized algebraic data types
19. Why Dependent Types Matter
20. Types and programming languages
```

## GPU Acceleration
GPU accelerating your model runs its performance bottlenecks on the GPU rather than the CPU.

There's no reason to instantiate the GPU models directly, instead you can simply instantiate the normal version of a supported model, and then use the `@gpu` macro to train it on the GPU:

```julia
corp = readcorp(:nsf)

model = LDA(corp, 16)
@time @gpu train!(model, iter=150, tol=0, chkelbo=151) # Let's time it as well to get an exact benchmark. 

# training...

# 156.591258 seconds (231.34 M allocations: 25.416 GiB, 37.82% gc time)
# On an Intel Iris Plus Graphics 640 1536 MB GPU.
```

This algorithm just crunched through a 16 topic 128,804 document topic model in *under* 3 minutes.

**Important:** Notice that we didn't check the ELBO at all during training.  While you can check the ELBO if you wish, it's recommended that you do so infrequently since checking the ELBO for GPU models requires expensive transfers between GPU and CPU memory.

Here is the benchmark of our above model against the equivalent NSF LDA model run on the CPU:
![GPU Benchmark](https://github.com/esproff/TopicModelsVB.jl/blob/master/images/ldabar2.png)

As we can see, running the LDA model on the GPU is approximatey 1.32 orders of magnitude faster than running it on the CPU.

It's often the case that one does not have sufficient VRAM to hold the entire model in GPU memory at one time.  Thus we provide the option of batching GPU models in order to train much larger models than would otherwise be possible:

```julia
corp = readcorp(:citeu)

model = CTM(corp, 7)
@gpu 4250 train!(model, iter=150, chkelbo=25) # batchsize = 4250 documents.

# training...
```

It's important to understand that GPGPU is still the wild west of computer programming.  The performance of batched models depends on many architecture dependent factors, including but not limited to the memory, the GPU, the manufacturer, the type of computer, what other applications are running, whether a display is connected, etc.

While non-batched models will usually be the fastest (for those GPUs which can handle them), it's not necessarily the case that reducing the batch size will result in a degredation in performance.  Thus it's always a good idea to experiment with different batch sizes, to see which sizes work best for your computer.

**Important:** If Julia crashes or throws an error when trying to run one of the models on your GPU, then your best course of action is to reduce the batch size and retrain your model.

Finally, expect your computer to lag when training on the GPU, since you're effectively siphoning off its rendering resources to fit your model.

## Types

```julia
VectorList{T}
# Array{Array{T,1},1}

MatrixList{T}
# Array{Array{T,2},1}

Document(terms::Vector{Integer}; counts::Vector{Integer}=ones(length(terms)), readers::Vector{Integer}=Int[], ratings::Vector{Integer}=ones(length(readers)), stamp::Real=-Inf, title::UTF8String="")
# FIELDNAMES:
# terms::Vector{Int}
# counts::Vector{Int}
# readers::Vector{Int}
# ratings::Vector{Int}
# stamp::Float64
# title::UTF8String

Corpus(;docs::Vector{Document}=Document[], lex::Union{Vector{UTF8String}, Dict{Integer, UTF8String}}=[], users::Union{Vector{UTF8String}, Dict{Integer, UTF8String}}=[])
# FIELDNAMES:
# docs::Vector{Document}
# lex::Dict{Int, UTF8String}
# users::Dict{Int, UTF8String}

TopicModel
# abstract

GPUTopicModel <: TopicModel
# abstract

BaseTopicModel
# Union{LDA, fLDA, CTM, fCTM, gpuLDA, gpuCTM}

AbstractLDA
# Union{LDA, gpuLDA}

AbstractfLDA
# Union{fLDA}

AbstractCTM
# Union{CTM, gpuCTM}

AbstractfCTM
# Union{fCTM}

AbstractDTM
# Union{DTM}

AbstractCTPF
# Union{CTPF, gpuCTPF}

LDA(corp::Corpus, K::Integer) <: TopicModel
# Latent Dirichlet allocation
# 'K' - number of topics.

fLDA(corp::Corpus, K::Integer) <: TopicModel
# Filtered latent Dirichlet allocation

CTM(corp::Corpus, K::Integer) <: TopicModel
# Correlated topic model

fCTM(corp::Corpus, K::Integer) <: TopicModel
# Filtered correlated topic model

DTM(corp::Corpus, K::Integer, delta::Real, basemodel::BaseTopicModel) <: TopicModel
# Dynamic topic model
# 'delta'     - time-interval size.
# 'basemodel' - pre-trained model of type BaseTopicModel (optional).

CTPF(corp::Corpus, K::Integer, basemodel::BaseTopicModel) <: GPUTopicModel
# Collaborative topic Poisson factorization
# 'basemodel' - pre-trained model of type BaseTopicModel (optional).

gpuLDA(corp::Corpus, K::Integer, batchsize::Integer) <: GPUTopicModel
# GPU accelerated latent Dirichlet allocation
# 'batchsize' defaults to 'length(corp)'.

gpuCTM(corp::Corpus, K::Integer, batchsize::Integer) <: GPUTopicModel
# GPU accelerated correlated topic model
# 'batchsize' defaults to 'length(corp)'.

gpuCTPF(corp::Corpus, K::Integer, batchsize::Integer, basemodel::BaseTopicModel) <: GPUTopicModel
# GPU accelerated collaborative topic Poission factorization
# 'batchsize' defaults to 'length(corp)'.
# 'basemodel' - pre-trained model of type BaseTopicModel (optional).
```

## Functions
### Generic Functions

```julia
isnegative(x::Union{Real, Array{Real}})
# Take Real or Array{Real} and return Bool or Array{Bool} (resp.).

ispositive(x::Union{Real, Array{Real}})
# Take Real or Array{Real} and return Bool or Array{Bool} (resp.).

logsumexp(x::Array{Real})
# Overflow safe log(sum(exp(x))).

addlogistic(x::Array{Real}, region::Integer)
# Overflow safe additive logistic function.
# 'region' is optional, across columns: 'region' = 1, rows: 'region' = 2.

partition(xs::Union{Vector, UnitRange}, n::Integer)
# 'n' must be positive.
# Return VectorList containing contiguous portions of xs of length n (includes remainder).
# e.g. partition([1,-7.1,"HI",5,5], 2) == Vector[[1,-7.1],["HI",5],[5]]
```

### Document/Corpus Functions
```julia
checkdoc(doc::Document)
# Verify that all Document fields have legal values.

checkcorp(corp::Corpus)
# Verify that all Corpus fields have legal values.

readcorp(;docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Read corpus from plaintext files.
# readcorp(:nsf)   - National Science Foundation corpus.
# readcorp(:citeu) - CiteULike corpus.
# readcorp(:mac)   - Macintosh Magazine corpus.

writecorp(corp::Corpus; docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Write corpus to plaintext files.

abridgecorp!(corp::Corpus; stop::Bool=false, order::Bool=true, abr::Integer=1)
# Abridge corpus.
# If 'stop' = true, stop words are removed.
# If 'order' = false, order is ignored and multiple seperate occurrences of words are stacked and the associated counts increased.
# All terms which appear < 'abr' times are removed from documents.

trimcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true)
# Those values which appear in the indicated fields of documents, yet don't appear in the corpus dictionaries, are removed.

compactcorp!(corp::Corpus; lex::Bool=true, users::Bool=true, alphabetize::Bool=true)
# Compact a corpus by relabeling lex and/or userkeys so that they form a unit range.
# If alphabetize=true the lex and/or user dictionaries are alphabetized.

padcorp!(corp::Corpus; lex::Bool=true, users::Bool=true)
# Pad a corpus by entering generic values for lex and/or userkeys which appear in documents but not in the lex/user dictionaries.

cullcorp!(corp::Corpus; lex::Bool=false, users::Bool=false, len::Integer=1)
# Culls the corpus of documents which contain lex and/or user keys in a document's terms/readers (resp.) fields yet don't appear in the corpus dictionaries.
# All documents of length < len are removed.

fixcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true, stop::Bool=false, order::Bool=true, abr::Int=1, len::Int=1, alphabetize::Bool=true)
# Fixes a corp by running the following four functions in order:
# abridgecorp!(corp, stop=stop, order=order, abr=abr)
# trimcorp!(corp, lex=lex, terms=terms, users=users, readers=readers)
# cullcorp!(corp, len=len)	
# compactcorp!(corp, lex=lex, users=users, alphabetize=alphabetize)

showdocs(corp::Corpus, docs::Union{Document, Vector{Document}, Integer, Vector{Integer}, UnitRange{Integer}})
# Display the text and title of a document(s).

getlex(corp::Corpus)
# Collect sorted values from the lex dictionary.

getusers(corp::Corpus)
# Collect sorted values from the user dictionary.
```

### Model Functions

```julia
showdocs(model::TopicModel, docs::Union{Document, Vector{Document}, Int, Vector{Int}, UnitRange{Int}})
# Display the text and title of a document(s).

fixmodel!(model::TopicModel; check::Bool=true)
# If 'check == true', verify the legality of the model's primary data.
# Align any auxiliary parameters with their associated parent parameters.

train!(model::BaseTopicModel; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
# Train one of the following models: LDA, fLDA, CTM, fCTM.
# 'iter'    - maximum number of iterations through the corpus.
# 'tol'     - absolute tolerance for ∆elbo as a stopping criterion.
# 'niter'   - maximum number of iterations for Newton's and interior-point Newton's methods.
# 'ntol'    - tolerance for change in function value as a stopping criterion for Newton's and interior-point Newton's methods.
# 'viter'   - maximum number of iterations for optimizing variational parameters (at the document level).
# 'vtol'    - tolerance for change in variational parameter values as stopping criterion.
# 'chkelbo' - number of iterations between ∆elbo checks (for both evaluation and convergence of the evidence lower-bound).

train!(dtm::AbstractDTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/dtm.K^2, cgiter::Integer=10, cgtol::Real=1/dtm.T^2, chkelbo::Integer=1)
# Train DTM.
# 'cgiter' - maximum number of iterations for the Polak-Ribière conjugate gradient method.
# 'cgtol'  - tolerance for change in function value as a stopping criterion for the Polak-Ribière conjugate gradient method.

train!(ctpf::AbstractCTPF; iter::Integer=150, tol::Real=1.0, viter::Integer=10, vtol::Real=1/ctpf.K^2, chkelbo::Integer=1)
# Train CTPF.

@gpu train!(model; kwargs...)
# Train model on GPU.

gendoc(model::BaseTopicModel, a::Real=0.0)
# Generate a generic document from model parameters by running the associated graphical model as a generative process.
# 'a' - amount of Laplace smoothing to apply to the topic-term distributions ('a' must be nonnegative).

gencorp(model::BaseTopicModel, corpsize::Int, a::Real=0.0)
# Generate a generic corpus of size 'corpsize' from model parameters.

showtopics(model::TopicModel, N::Integer=min(15, model.V); topics::Union{Integer, Vector{Integer}}=collect(1:model.K), cols::Integer=4)
# Display the top 'N' words for each topic in 'topics', defaults to 4 columns per line.

showtopics(dtm::AbstractDTM, N::Integer=min(15, dtm.V); topics::Union{Integer, Vector{Integer}}=collect(1:dtm.K), times::Union{Integer, Vector{Integer}}=collect(1:dtm.T), cols::Integer=4)
# Display the top 'N' words for each topic in 'topics' and each time interval in 'times', defaults to 4 columns per line.

showlibs(ctpf::AbstractCTPF, users::Union{Integer, Vector{Integer}})
# Show the document(s) in a user's library.

showdrecs(ctpf::AbstractCTPF, docs::Union{Integer, Vector{Integer}}, U::Integer=min(16, ctpf.U); cols::Integer=4)
# Show the top 'U' user recommendations for a document(s), defaults to 4 columns per line.

showurecs(ctpf::AbstractCTPF, users::Union{Integer, Vector{Integer}}, M::Integer=min(10, ctpf.M); cols::Integer=1)
# Show the top 'M' document recommendations for a user(s), defaults to 1 column per line.
# If a document has no title, the document's index in the corpus will be shown instead.
```

## Bibliography
1. Latent Dirichlet Allocation (2003); Blei, Ng, Jordan. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf)
2. Filtered Latent Dirichlet Allocation: Variational Bayes Algorithm (2016); Proffitt. [pdf](https://github.com/esproff/TopicModelsVB.jl/blob/master/fLDAVB.pdf)
3. Correlated Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006.pdf)
4. Dynamic Topic Models (2006); Blei, Lafferty. [pdf](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2006a.pdf)
5. Content-based Recommendations with Poisson Factorization (2014); Gopalan, Charlin, Blei. [pdf](http://www.cs.columbia.edu/~blei/papers/GopalanCharlinBlei2014.pdf)
6. Numerical Optimization (2006); Nocedal, Wright. [Amazon](https://www.amazon.com/Numerical-Optimization-Operations-Financial-Engineering/dp/0387303030)
7. Machine Learning: A Probabilistic Perspective (2012); Murphy. [Amazon](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr=)
8. OpenCL in Action: How to Accelerate Graphics and Computation (2011); Scarpino. [Amazon](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173)
