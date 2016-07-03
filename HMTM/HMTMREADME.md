# Hidden Markov Topic Model

The hidden markov topic model (HMTM), is a topic model which takes word order into account.

The original paper (which uses Gibbs sampling), can be found here: http://www.mjandrews.net/papers/andrews.cogsci.2009.pdf

I designed a variational Bayes algorithm for HMTM, but was unable to update one of the coordinates, a detailed description of the algorithm and precisely where I became stuck is included in the HMTMVB pdf.

In order to manually integrate the HMTM.jl file into the TopicModelsVB package, so that you can try to complete this algorithm yourself, all you need to do is to put the HMTM.jl file in the path:

**~/.julia/v0.4/topicmodelsvb/src/HMTM.jl**

Then the final thing you need to do is open up the TopicModelsVB.jl file in the path:

**~/.julia/v0.4/topicmodelsvb/src/TopicModelsVB.jl**

and both add `include("HMTM.jl")` to the collection of other files at the bottom, and then add `HMTM` to the list of models on the first export line.

Also included in this HMTM folder are doc, lex and title files for a ~12k document dataset of articles from *PC Today Magazine* 2004 - 2012.  The order of words is preserved in this dataset, and stopwords have *not* been removed.  Remember you can read the articles with the function `showdocs`.

Now you can run the algorithm on a corpus as is, however the `updatePhi!` function you'll notice is empty, since this is the coordinate I was unable to optimize (see the HMTMVB pdf).

I suspect that the only way to efficiently update `phi` is to lower-bound the portions of the objective function which contain `phi` with a function that can be optimized analytically, however finding this lower bound is going to be challenging.

Good luck!
