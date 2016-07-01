module TopicModelsVB

using Distributions

export VectorList, MatrixList, Document, Corpus, TopicModel, LDA, fLDA, CTM, fCTM, DTM, CTPF
export isnegative, ispositive, tetragamma, logsumexp, addlogistic, partition
export checkdoc, checkcorp, readcorp, writecorp, abridgecorp!, trimcorp!, compactcorp!, padcorp!, cullcorp!, fixcorp!, showdocs, getlex, getusers
export train!
export checkmodel, gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs

include("utils.jl")
include("Corpus.jl")
include("TopicModel.jl")
include("LDA.jl")
include("fLDA.jl")
include("CTM.jl")
include("fCTM.jl")
include("DTM.jl")
include("CTPF.jl")
include("modelutils.jl")

end
