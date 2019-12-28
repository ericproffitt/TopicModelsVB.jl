module TopicModelsVB

using DelimitedFiles
using Crayons
using Random
using SpecialFunctions
using LinearAlgebra
using Distributions
using OpenCL

export Document, Corpus
export TopicModel
export LDA, fLDA, CTM, fCTM, CTPF, gpuLDA, gpuCTM, gpuCTPF
export readcorp, writecorp, abridge_corp!, alphabetize_corp!, compact_corp!, condense_corp!, pad_corp!, trim_corp!, remove_empty_docs!, stop_corp!, trim_docs!, fixcorp!, showdocs, getlex, getusers
export train!
export @gpu
export gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs

include("macros.jl")
include("utils.jl")
include("Corpus.jl")
include("TopicModel.jl")
include("LDA.jl")
include("fLDA.jl")
include("CTM.jl")
include("fCTM.jl")
include("CTPF.jl")
include("gpuLDA.jl")
include("gpuCTM.jl")
include("gpuCTPF.jl")
include("modelutils.jl")

end
