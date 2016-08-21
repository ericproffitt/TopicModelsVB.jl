# TopicModelsVB.jl
A Julia Package for Variational Bayesian Topic Modeling.

Topic Modeling is concerned with discovering the latent low-dimensional thematic structure within corpora.  Modeling this latent structure is done using either [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC) methods, or [variational Bayesian](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) (VB) methods.  The former approach is slower, but unbiased.  Given infinite time, MCMC will fit the desired model exactly.  The latter method is faster (often much faster), but biased, since one must approximate distributions in order to ensure tractability.  This package takes the latter approach to topic modeling.

## Dependencies

```julia
Pkg.add("Distributions")
Pkg.add("OpenCL")
```

## Install

```julia
Pkg.clone("git://github.com/esproff/TopicModelsVB.jl.git")
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

* `terms`: A line of delimited positive integers corresponding to the terms which make up the document (this line is mandatory).

* `counts`: A line of delimited positive integers, equal in length to the term line, corresponding to the number of times a particular term appears in a document (defaults to `ones(length(terms))`).

* `readers`: A line of delimited positive integers corresponding to those users which have read the document.

* `ratings`: A line of delimited positive integers, equal in length to the `readers` line, corresponding to the rating each reader gave the document (defaults to `ones(length(readers))`).

* `stamp`: A numerical value in the range `[-inf, inf]` denoting the timestamp of the document.

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

The lex and user files are dictionaries mapping positive integers to terms and usernames (resp.).  For example,

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

Whenever you load a corpus into a model, a copy of that corpus is made, such that if you modify the original corpus at corpus-level (remove documents, re-order lex keys, etc.), this will not affect any corpus attached to a model.  However!  Since corpora are containers for their documents, modifying an individual document will affect this document in all corpora which contain it.  **Be very careful whenever modifying the internals of documents themselves, either manually or through the use of** `corp!` **functions**. 

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

gpufLDA(corp, K, batchsize)
# Coming soon...

gpuCTM(corp, K, batchsize)
# GPU accelerated correlated topic model with K topics.

gpufCTM(corp, K, batchsize)
# Coming soon...

gpuDTM(corp, K, delta, batchsize, basemodel)
# Coming soon...

gpuCTPF(corp, K, batchsize, basemodel)
# GPU accelerated collaborative topic Poisson factorization model with K topics.
```

## Tutorial
### LDA
Let's begin our tutorial with a simple latent Dirichlet allocation (LDA) model with 9 topics, trained on the first 5000 documents from the NSF corpus.

```julia
using TopicModelsVB

srand(1)

nsfcorp = readcorp(:nsf) 

nsfcorp.docs = nsfcorp[1:5000]
fixcorp!(nsfcorp)

# Notice that the post-fix lexicon is smaller after removing all but the first 5000 docs.

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

Now that we've trained our LDA model we can, if we want, take a look at the topic proportions for individual documents.  For instance, document 1 has topic breakdown:

```julia
nsflda.gamma[1] # = [0.036, 0.030, 189.312, 0.036, 0.049, 0.022, 8.728, 0.027, 0.025]
```
This vector of topic weights suggests that document 1 is mostly about biology, and in fact looking at the document text confirms this observation:

```julia
showdocs(nsflda, 1) # Could also have done showdocs(nsfcorp, 1).
```

```
 ●●● Doc: 1
 ●●● CRB: Genetic Diversity of Endangered Populations of Mysticete Whales: Mitochondrial DNA and Historical Demography
commercial exploitation past hundred years great extinction variation sizes
populations prior minimal population size current permit analyses effects 
differing levels species distributions life history...
```

On the other hand, some documents will be a combination of topics.  Consider the topic breakdown for document 25:

```julia
nsflda.gamma[25] # = [11.575, 44.889, 0.0204, 0.036, 0.049, 0.022, 0.020, 66.629, 0.025]

showdocs(nsflda, 25)
```

```
 ●●● Doc: 25
 ●●● Mathematical Sciences: Nonlinear Partial Differential Equations from Hydrodynamics
work project continues mathematical research nonlinear elliptic problems arising perfect
fluid hydrodynamics emphasis analytical study propagation waves stratified media techniques
analysis partial differential equations form basis studies primary goals understand nature 
internal presence vortex rings arise density stratification due salinity temperature...
```

We see that in this case document 25 appears to be about applications of mathematical physics to ocean currents, which corresponds precisely to a combination of topics 2 and 8, with a smaller but not insignificant weight on topic 1.

Furthermore, if we want to, we can also generate artificial corpora by using the ```gencorp``` function.  Generating artificial corpora will in turn run the underlying probabilistic graphical model as a generative process in order to produce entirely new collections of documents, let's try it out:

```julia
artifnsfcorp = gencorp(nsflda, 5000, 1e-5) # The third argument governs the amount of Laplace smoothing (defaults to 0.0).

artifnsflda = LDA(artifnsfcorp, 9)
train!(artifnsflda, iter=150, tol=0.0, chkelbo=15)

# training...

showtopics(artifnsflda, cols=9)
```

```
topic 1       topic 2          topic 3       topic 4          topic 5         topic 6         topic 7      topic 8         topic 9
cell          research         research      species          theory          data            chemistry    research        research
protein       project          university    plant            problems        project         research     study           systems
cells         study            students      research         study           research        reactions    systems         design
gene          data             support       study            research        earthquake      metal        phase           system
proteins      economic         program       evolutionary     equations       study           chemical     experimental    data
plant         social           science       genetic          work            studies         organic      flow            algorithms
studies       important        scientists    population       project         water           structure    theoretical     based
genes         understanding    scientific    plants           investigator    ocean           program      materials       parallel
research      work             award         populations      principal       measurements    study        high            performance
molecular     information      sciences      dr               geometry        program         dr           quantum         techniques
specific      theory           projects      data             differential    important       molecular    physics         computer
mechanisms    provide          dr            patterns         mathematical    time            synthesis    properties      problems
system        development      project       relationships    algebraic       models          compounds    temperature     control
role          human            national      evolution        methods         seismic         surface      dynamics        project
study         political        provide       variation        analysis        field           studies      proposed        methods
```

One thing we notice so far is that despite producing what are clearly coherent topics, many of the top words in each topic are words such as *research*, *study*, *data*, etc.  While such terms would be considered informative in a generic corpus, they are effectively stop words in a corpus composed of science article abstracts.  Such corpus-specific stop words will be missed by most generic stop word lists, and they can be difficult to pinpoint and individually remove prior to training.  Thus let's change our model to a *filtered* latent Dirichlet allocation (fLDA) model.

```julia
srand(1)

nsfflda = fLDA(nsfcorp, 9)
train!(nsfflda, iter=150, tol=0.0)

# training...

showtopics(nsfflda, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4          topic 5        topic 6       topic 7      topic 8         topic 9
earthquake      flow            species          design           university     cell          economic     theory          chemistry
ocean           theoretical     plant            algorithms       support        protein       social       equations       reactions
water           phase           populations      computer         students       cells         theory       geometry        metal
measurements    physics         genetic          performance      program        proteins      policy       algebraic       chemical
program         quantum         plants           parallel         science        gene          human        differential    program
soil            properties      evolutionary     processing       award          plant         change       mathematical    organic
seismic         temperature     population       applications     scientists     genes         political    groups          molecular
climate         effects         patterns         networks         scientific     molecular     public       space           compounds
effects         phenomena       variation        network          sciences       function      examine      mathematics     surface
global          numerical       effects          software         conference     dna           science      finite          properties
sea             laser           food             computational    national       regulation    decision     solutions       molecules
surface         measurements    ecology          efficient        projects       expression    people       spaces          university
response        experiments     environmental    program          engineering    plants        labor        dimensional     reaction
solar           award           test             distributed      year           mechanisms    effects      functions       synthesis
earth           liquid          ecological       power            workshop       membrane      market       questions       complexes
```

We can now see that many of the most troublesome corpus-specific stop words have been automatically filtered out, while those that remain are those which tend to cluster within their own, more generic, topic.

### CTM
For our final example using the NSF corpus, let's upgrade our model to a filtered *correlated* topic model (fCTM).

```julia
srand(1)

nsffctm = fCTM(nsfcorp, 9)
train!(nsffctm, iter=150, tol=0.0)

# training...

showtopics(nsffctm, 20, cols=9)
```

```
topic 1         topic 2         topic 3          topic 4          topic 5         topic 6       topic 7      topic 8         topic 9
data            flow            species          system           university      cell          data         theory          chemistry
earthquake      numerical       plant            design           support         protein       social       problems        chemical
ocean           theoretical     populations      data             program         cells         economic     geometry        materials
water           models          genetic          algorithms       students        plant         theory       investigator    reactions
measurements    model           evolutionary     control          science         proteins      policy       algebraic       properties
program         physics         plants           problems         dr              gene          human        equations       metal
climate         theory          population       models           award           molecular     political    groups          surface
models          nonlinear       data             parallel         scientists      genes         models       differential    electron
seismic         dynamics        dr               computer         scientific      dna           public       space           program
soil            experimental    patterns         performance      sciences        system        change       mathematical    molecular
earth           equations       relationships    model            national        function      model        mathematics     organic
global          particle        evolution        processing       projects        regulation    science      spaces          dr
sea             phenomena       variation        applications     engineering     plants        people       functions       university
response        quantum         group            network          conference      expression    decision     questions       compounds
damage          heat            ecology          networks         year            mechanisms    issues       manifolds       temperature
solar           fluid           ecological       approach         researchers     dr            labor        finite          molecules
pacific         particles       forest           software         workshop        membrane      market       dimensional     laser
ice             waves           environmental    efficient        mathematical    genetic       case         properties      reaction
surface         problems        food             computational    months          binding       women        group           optical
system          award           experiments      distributed      equipment       cellular      factors      operators       measurements
```

Because the topics in the fLDA model were already so well defined, there's little room to improve topic coherence by upgrading to the fCTM model, however what's most interesting about the CTM and fCTM models is the ability to look at topic correlations.

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

Now let's take a look at the topic-covariance matrix:

```julia
nsffctm.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
nsffctm.sigma[4,8] # 9.532
nsffctm.sigma[3,6] # 7.362
nsffctm.sigma[2,9] # 4.531

# Top 3 negative entries, sorted in ascending order:
nsffctm.sigma[7,9] # -14.627
nsffctm.sigma[3,8] # -12.464
nsffctm.sigma[1,8] # -11.775
```

According to the list above, the most closely related topics are topics 4 and 8, which correspond to the *Computer Science* and *Mathematics* topics, followed closely by 3 and 6, corresponding to the topics *Sociobiology* and *Microbiology*, and then by 2 and 9, corresponding to *Physics* and *Chemistry*.

As for the most unlikely topic pairings, first are topics 7 and 9, corresponding to *Economics* and *Chemistry*, followed closely by topics 1 and 8, corresponding to *Sociobiology* and *Mathematics*, and then third are topics 3 and 8, corresponding to *Earth Science* and *Mathematics*.

Furthermore, as expected, the topic which is least correlated with all other topics is the *Academia* topic:

```julia
sum(abs(nsffctm.sigma[:,5])) - nsffctm.sigma[5,5] # Academia topic, absolute off-diagonal covariance 13.403.
```

**Note:** Both CTM and fCTM will sometimes have to numerically invert ill-conditioned matrices, thus don't be alarmed if the ```∆elbo``` periodically goes negative for stretches, it should always right itself in fairly short order.

### DTM
Now that we have covered static topic models, let's transition to the dynamic topic model (DTM).  The dynamic topic model discovers the temporal-dynamics of topics which, nevertheless, remain thematically static.  A good example of a topic which is thematically-static, yet exhibits an evolving lexicon, is *Computer Storage*.  Methods of data storage have evolved rapidly in the last 40 years, evolving from punch cards, to 5-inch floppy disks, to smaller hard disks, to zip drives and cds, to dvds and platter hard drives, and now to flash drives, solid-state drives and cloud storage, all accompanied by the rise and fall of computer companies which manufacture (or at one time manufactured) these products.

As an example, let's load the Macintosh corpus of articles, drawn from the magazines *MacWorld* and *MacAddict*, published between the years 1984 - 2005.  We sample 400 articles randomly from each year, and break time periods into 2 year intervals.

```julia
import Distributions.sample

srand(1)

maccorp = readcorp(:mac)

maccorp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, maccorp.docs), 400, replace=false) for y in 1984:2005]...)

fixcorp!(maccorp, b=100, len=10) # Remove words which appear < 100 times and documents of length < 10.

basemodel = LDA(maccorp, 9)
train!(basemodel, iter=150, chkelbo=151)

# training...

macdtm = DTM(maccorp, 9, 200, pmodel)
train!(macdtm, iter=10) # This will likely take about an hour on a personal computer.
                        # Convergence for all other models is worst-case quadratic,
                        # while DTM convergence is linear or at best super-linear.
# training...
```

We can look at a particular topic slice by writing:

```julia
showtopics(macdtm, topics=4, cols=6)
```

```
 ●●● Topic: 4
time 1       time 2       time 3         time 4       time 5       time 6
board        macintosh    color          board        board        power
serial       upgrade      system         video        powerbook    macs
memory       port         board          ram          color        price
power        memory       video          upgrade      upgrade      video
upgrade      board        display        system       display      speed
port         power        memory         rom          apple        upgrade
chips        expansion    boards         simms        scsi         performance
unit         unit         radius         macintosh    video        apple
ports        digital      upgrade        ethernet     monitor      fast
chip         connect      power          classic      power        monitors
expansion    radius       ram            apple        quadra       powerbook
digital      device       monitor        math         memory       radius
boards       devices      accelerator    rasterops    designed     small
plug         boards       device         scsi         processor    cpu
adapter      chip         supermac       digital      macintosh    faster

time 7         time 8         time 9         time 10      time 11       
power          g3             usb            g4           ipod          
ram            usb            firewire       power        mini          
speed          imac           g4             usb          power         
processor      ram            ram            firewire     usb           
standard       apple          apple          apple        apple         
fast           modem          power          ram          g5            
drive          upgrade        palm           imac         firewire      
performance    power          g3             powerbook    g4            
keyboard       pci            refurbished    drive        ram           
faster         speed          powerbook      airport      models        
upgrade        drive          hardware       port         display       
memory         powerbook      machine        processor    memory        
apple          internal       port           models       hard_drive    
slots          serial         imac           memory       speed         
powerbook      performance    device         pci          port  
```

or a particular time slice, by writing:

```julia
showtopics(macdtm, times=11, cols=9)
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

As you may have noticed, the dynamic topic model is *extremely* computationally intensive, hopefully GPGPU support will ameliorate this problem to at least some degree.  However it is likely that running the DTM model on an industry-sized dataset will always require more computational power than can be provided by your standard personal computer.

**Important:** Beware that the DTM algorithm is still a bit buggy, an overhaul of the algorithm itself will likely come alongside the GPU accelerated version.

### CTPF
For our final model, we take a look at the collaborative topic Poisson factorization (CTPF) model.  CTPF is a collaborative filtering topic model which uses the latent thematic structure of documents to improve the quality of document recommendations beyond what would be achievable using just the document-user matrix.  This blending of thematic structure with known user prefrences not only improves recommendation accuracy, but also mitigates the cold-start problem of recommending to users never-before-seen documents.  As an example, let's load the CiteULike dataset into a corpus and then randomly remove a single reader from each of the documents.

```julia
import Distributions.sample

srand(1)

citeucorp = readcorp(:citeu)

testukeys = Int[]
for doc in citeucorp
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
sum([isempty(doc.readers) for doc in citeucorp]) # = 158
```

Fortunately, since CTPF can if need be depend entirely on thematic structure when making recommendations, this poses no problem for the model.

Now that we have set up our experiment, we instantiate and train a CTPF model on our corpus.  Furthermore, since we're not interested in the interpretability of the topics, we'll instantiate our model with a larger than usual number of topics (K=30), and then run it for a relatively short number of iterations (iter=20).

```julia
srand(1)

citeuctpf = CTPF(citeucorp, 30) # Note: If no 'pmodel' is entered then parameters will be initialized at random.
train!(citeuctpf, iter=20)

# training...
```

Finally, we evaluate the accuracy of our model against the test set, where baseline for mean accuracy is 0.5.

```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)[1]
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.908
```

Not bad, but let's see if we can't improve our accuracy at least a percentage point or two by priming our CTPF model with a 100 iteration LDA model.

In the interest of time, let's use the GPU accelerated verions of LDA and CTPF:


```julia
srand(1)

basemodel = gpuLDA(citeucorp, 30)
train!(basemodel, iter=100, chkelbo=101)

# training...

citeuctpf = gpuCTPF(citeucorp, 30, basemodel)
train!(citeuctpf, iter=20, chkelbo=21)

# training...
```

Again we evaluate the accuracy of our model against the test set:

```julia
acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)[1]
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.920
```

We can see that, on average, our model ranks the true hidden reader in the top 8% of all non-readers for each document.

Let's also take a look at the top recommendations for a particular document(s):

```julia
testukeys[1] # = 216
acc[1] # = 0.945

showdrecs(citeuctpf, 1, 307, cols=1)
```
```
 ●●● Doc: 1
 ●●● The metabolic world of Escherichia coli is not small
...
303. #user5159
304. #user5486
305. #user261
306. #user4999
307. #user216
```
as well as those for a particular user(s):

```julia
showurecs(citeuctpf, 216, 1745)
```
```
 ●●● User: 216
...
1741. {Characterizing gene sets with FuncAssociate}
1742. Network Data and Measurement
1743. Analysis of genomic context: prediction of functional associations from conserved bidirectionally transcribed gene pairs.
1744. The public road to high-quality curated biological pathways
1745. The metabolic world of Escherichia coli is not small
```

We can also take a more holistic and informal approach to evaluating model quality.

Since large heterogenous libraries make the qualitative assessment of recommendations difficult, let's search for a user with a modestly sized relatively focused library: 
```julia
showlibs(citeuctpf, 1741)
```

```
 ●●● User: 1741
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
 
 The 20 articles in user 1741's library suggest that he or she is interested in the interplay of foundational mathematics and functional programming.  
 
 Now compare this with the top 20 recommendations made by our model:
 
```julia
showurecs(citeuctpf, 1741, 20)
```

```
 ●●● User: 1741
1.  Sets for Mathematics
2.  On Understanding Types, Data Abstraction, and Polymorphism
3.  Can programming be liberated from the von {N}eumann style? {A} functional style and its algebra of programs
4.  Dynamic Logic
5.  Principles of programming with complex objects and collection types
6.  Functional programming with bananas, lenses, envelopes and barbed wire
7.  Haskell's overlooked object system
8.  Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints
9.  Lectures on the Curry-Howard isomorphism
10. Parsing expression grammars: a recognition-based syntactic foundation
11. Type Classes with Functional Dependencies
12. Macros as multi-stage computations: type-safe, generative, binding macros in MacroML
13. Types, abstraction and parametric polymorphism
14. Functional pearl: implicit configurations--or, type classes reflect the values of types
15. The Zipper
16. Monadic Parsing in Haskell
17. Ownership types for safe programming: preventing data races and deadlocks
18. Monadic Parser Combinators
19. The essence of compiling with continuations
20. The {Calculus of Constructions}
```

## GPU Acceleration
GPU accelerating your model runs its performance bottlenecks on the GPU rather than the CPU.

Currently the LDA, CTM and CTPF models are supported, however GPU accelerated versions of the remaining three models are in the works.  There's no reason to instantiate the GPU models directly, instead you can simply instantiate the normal version of a supported model, and then use the `@gpu` macro to train it on the GPU:

```julia
nsfcorp = readcorp(:nsf)

nsflda = LDA(nsfcorp, 16)
@time @gpu train!(nsflda, iter=150, chkelbo=151) # Let's time it as well to get an exact benchmark. 

# training...

# 238.117185 seconds (180.46 M allocations: 11.619 GB, 4.39% gc time)
# On a 2.5 GHz Intel Core i5 2012 Macbook Pro with 4GB of RAM and an Intel HD Graphics 4000 1536 MB GPU.
```

This algorithm just crunched through a 16 topic 128,804 document topic model in *under* 4 minutes.

**Important:** Notice that we didn't check the ELBO at all during training.  While you can check the ELBO if you wish, it's recommended that you do so infrequently since checking the ELBO for GPU models requires expensive transfers between GPU and CPU memory.

Here is the benchmark of our above model against the equivalent NSF LDA model run on the CPU:
![GPU Benchmark](https://github.com/esproff/TopicModelsVB.jl/blob/master/images/ldabar.png)

As we can see, the GPU LDA model is approximatey 1.35 orders of magnitude faster than the equivalent CPU LDA model.

It's often the case that one does not have sufficient VRAM to hold the entire GPU model at one time.  Thus we provide the option of batching the GPU model in order to fit much larger models than would otherwise be possible:

```julia
citeucorp = readcorp(:citeu)

citeuctm = CTM(citeucorp, 7)
@gpu 4250 train!(citeuctm, iter=150, chkelbo=25) # batchsize = 4250 documents.

# training...
```

It's important to understand that GPGPU is still the wild west of computer programming.  The performance of batched models depends on many architecture dependent factors, including but not limited to the memory, the GPU, the manufacturer, the type of computer, what other applications are running, whether a display is connected, etc.

While non-batched models will usually be the fasted (for those GPUs which can handle them), it's not necessarily the case that reducing the batch size will result in a degredation in performance.  Thus it's always a good idea to experiment with different batch sizes, to see which sizes work best for your computer.

**Important:** If Julia crashes or throws an error when trying to run one of your models on the GPU, your best course of action is to reduce the batch size and retrain your model.

Finally, expect your computer to lag when training on your GPU, since you're effectively siphoning off its rendering resources to fit your model.

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
# Union{LDA, fLDA, CTM, fCTM, gpuLDA, gpufLDA, gpuCTM, gpufCTM}

AbstractLDA
# Union{LDA, gpuLDA}

AbstractfLDA
# Union{fLDA, gpufLDA}

AbstractCTM
# Union{CTM, gpuCTM}

AbstractfCTM
# Union{fCTM, gpufCTM}

AbstractDTM
# Union{DTM, gpuDTM}

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

gpufLDA(corp::Corpus, K::Integer, batchsize::Integer) <: GPUTopicModel
# GPU accelerated filtered latent Dirichlet allocation
# 'batchsize' defaults to 'length(corp)'.

gpuCTM(corp::Corpus, K::Integer, batchsize::Integer) <: GPUTopicModel
# GPU accelerated correlated topic model
# 'batchsize' defaults to 'length(corp)'.

gpufCTM(corp::Corpus, K::Integer, batchsize::Integer) <: GPUTopicModel
# GPU accelerated filtered correlated topic model
# 'batchsize' defaults to 'length(corp)'.

gpuDTM(corp::Corpus, K::Integer, delta::Real, batchsize::Integer, basemodel::BaseTopicModel) <: GPUTopicModel
# GPU accelerated dynamic topic model
# 'delta'     - time-interval size.
# 'batchsize' defaults to 'length(corp)'.
# 'basemodel' - pre-trained model of type BaseTopicModel (optional).

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
# readcorp(:cmag)  - Full Computer Magazine corpus.

writecorp(corp::Corpus; docfile::AbstractString="", lexfile::AbstractString="", userfile::AbstractString="", titlefile::AbstractString="", delim::Char=',', counts::Bool=false, readers::Bool=false, ratings::Bool=false, stamps::Bool=false)
# Write corpus to plaintext files.

abridgecorp!(corp::Corpus; stop::Bool=false, order::Bool=true, abr::Integer=1)
# Abridge corpus.
# If stop = true, stop words are removed.
# If order = false, order is ignored and multiple seperate occurrences of words are stacked and the associated counts increased.
# All terms which appear < abr times are removed from documents.

trimcorp!(corp::Corpus; lex::Bool=true, terms::Bool=true, users::Bool=true, readers::Bool=true)
# Those values which appear in the indicated fields of documents, yet don't appear in the corpus dictionaries, are removed.

compactcorp!(corp::Corpus; lex::Bool=true, users::Bool=true, alphabetize::Bool=true)
# Compact a corpus by relabeling lex and/or userkeys so that they form a unit range.
# If alphabetize=true the lex and/or user dictionaries are alphabetized.

padcorp!(corp::Corpus; lex::Bool=true, users::Bool=true)
# Pad a corpus by entering generic values for lex and/or userkeys which appear in documents but not in the lex/user dictionaries.

cullcorp!(corp::Corpus; terms::Bool=false, readers::Bool=false, len::Integer=1)
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
# 'chkelbo' - number of iterations between ∆elbo checks (for both evaluation and convergence checking).

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
