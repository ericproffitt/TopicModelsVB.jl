module TopicModelsVB

using Random
using DelimitedFiles
using Crayons
using SpecialFunctions
using LinearAlgebra
using Distributions
using OpenCL

export VectorList, MatrixList
export Document, Corpus
export TopicModel, BaseTopicModel
export AbstractLDA, AbstractfLDA, AbstractCTM, AbstractfCTM, AbstractCTPF
export LDA, fLDA, CTM, fCTM, CTPF
export gpuLDA, gpufLDA, gpuCTM, gpufCTM, gpuCTPF
export logsumexp, additive_logistic
export readcorp, writecorp, abridge_corp!, alphabetize_corp!, compact_corp!, condense_corp!, pad_corp!, trim_corp!, remove_empty_docs!, stop_corp!, trim_docs!, fixcorp!, showdocs, getlex, getusers
export train!
export @gpu
export gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs

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

include("CTPF.jl")
include("gpuCTPF.jl")
include("AbstractCTPF.jl")

include("modelutils.jl")

end
