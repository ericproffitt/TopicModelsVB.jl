# Hidden Markov Topic Model

The hidden markov topic model (HMTM) is a topic model which takes word order into account.

The original paper (which uses Gibbs sampling), can be found here: http://www.mjandrews.net/papers/andrews.cogsci.2009.pdf

I designed a variational Bayes algorithm for HMTM, but was unable to update one of the coordinates, a detailed description of the algorithm and precisely where I became stuck is included in the [HMTMVB.pdf](https://github.com/esproff/TopicModelsVB.jl/blob/master/HMTM/HMTMVB.pdf).

Note that the HMTMVB pdf was originally written-up as supplementary notes for the author of the original paper.

In order to manually integrate the HMTM.jl file into the TopicModelsVB package, so that you can try to complete this algorithm yourself, all you need to do is first put the HMTM.jl file in the path:

**`~/.julia/v0.4/topicmodelsvb/src/HMTM.jl`**

then you need to open up the TopicModelsVB.jl file in the path:

**`~/.julia/v0.4/topicmodelsvb/src/TopicModelsVB.jl`**

and add both `include("HMTM.jl")` to the collection of other files at the bottom, and then add `HMTM` to the list of models on the first export line.

You can run the algorithm right away on the NSF dataset as is (the NSF dataset is ordered with stop words removed), however you'll notice that the `updatePhi!` function is empty, as this was the coordinate I was unable to optimize (see the HMTMVB pdf).

I suspect that the only way to efficiently update `phi` is to lower-bound the portions of the objective function which contain `phi` with a function that can be optimized analytically, however finding this lower bound is going to be challenging.

Also, the `updateLambda!` function could probably have its performance improved.

Good luck!
