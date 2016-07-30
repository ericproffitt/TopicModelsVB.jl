module TopicModelsVB

using Distributions
using OpenCL

export VectorList, MatrixList
export Document, Corpus
export TopicModel, BaseTopicModel, GPUTopicModel
export LDA, fLDA, CTM, fCTM, CTPF, DTM
export memLDA, memfLDA, memCTM, memfCTM, @mem
export gpuLDA, gpuCTPF
export isnegative, ispositive, tetragamma, logsumexp, addlogistic, partition
export checkdoc, checkcorp, readcorp, writecorp, abridgecorp!, trimcorp!, compactcorp!, padcorp!, cullcorp!, fixcorp!, showdocs, getlex, getusers
export train!, @gpu
export fixmodel!, gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs

include("macros.jl")
include("utils.jl")
include("Corpus.jl")
include("TopicModel.jl")
include("LDA.jl")
include("memLDA.jl")
include("gpuLDA.jl")
include("fLDA.jl")
include("memfLDA.jl")
include("CTM.jl")
include("memCTM.jl")
include("fCTM.jl")
include("memfCTM.jl")
include("BaseTopicModel.jl")
include("CTPF.jl")
include("gpuCTPF.jl")
include("DTM.jl")
include("GPUTopicModel.jl")
include("modelutils.jl")

end
