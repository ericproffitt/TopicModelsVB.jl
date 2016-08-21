module TopicModelsVB

using Distributions
using OpenCL

export VectorList, MatrixList
export Document, Corpus
export TopicModel, GPUTopicModel, BaseTopicModel
export AbstractLDA, AbstractfLDA, AbstractCTM, AbstractfCTM, AbstractDTM, AbstractCTPF
export LDA, fLDA, CTM, fCTM, CTPF, DTM
export gpuLDA, gpuCTM, gpuCTPF
export isnegative, ispositive, logsumexp, addlogistic, partition
export checkdoc, checkcorp, readcorp, writecorp, abridgecorp!, trimcorp!, compactcorp!, padcorp!, cullcorp!, fixcorp!, showdocs, getlex, getusers
export train!, @gpu
export fixmodel!, gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs

include("macros.jl")
include("utils.jl")
include("Corpus.jl")

include("TopicModel.jl")
include("GPUTopicModel.jl")

include("LDA.jl")
include("gpuLDA.jl")
include("AbstractLDA.jl")

include("fLDA.jl")
include("AbstractfLDA.jl")

include("CTM.jl")
include("gpuCTM.jl")
include("AbstractCTM.jl")

include("fCTM.jl")
include("AbstractfCTM.jl")

include("BaseTopicModel.jl")

include("DTM.jl")
include("AbstractDTM.jl")

include("CTPF.jl")
include("gpuCTPF.jl")
include("AbstractCTPF.jl")

include("modelutils.jl")

end
