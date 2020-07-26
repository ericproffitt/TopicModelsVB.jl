module TopicModelsVB

using DelimitedFiles
using SpecialFunctions
using LinearAlgebra
using Random
using Distributions
using OpenCL
using Crayons

export Document, Corpus
export TopicModel
export check_doc, check_corp, check_model
export LDA, fLDA, CTM, fCTM, CTPF, gpuLDA, gpuCTM, gpuCTPF
export readcorp, writecorp, abridge_corp!, alphabetize_corp!, compact_corp!, condense_corp!, pad_corp!, remove_empty_docs!, remove_redundant!, stop_corp!, trim_corp!, trim_docs!, fixcorp!, showdocs, showtitles, getvocab, getusers
export train!
export @gpu
export gendoc, gencorp, showtopics, showlibs, showdrecs, showurecs, predict, topicdist

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
