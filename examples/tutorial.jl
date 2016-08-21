using TopicModelsVB



###############################
#                             #
# Latent Dirichlet Allocation #
#                             #
###############################

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

nsflda.gamma[1] # = [0.036, 0.030, 189.312, 0.036, 0.049, 0.022, 8.728, 0.027, 0.025]

showdocs(nsflda, 1) # could also have done showdocs(nsfcorp, 1)

nsflda.gamma[25] # = [11.575, 44.889, 0.0204, 0.036, 0.049, 0.022, 0.020, 66.629, 0.025]

showdocs(nsflda, 25)

artifnsfcorp = gencorp(nsflda, 5000, 1e-5) # The third argument governs the amount of Laplace smoothing (defaults to 0.0).

artifnsflda = LDA(artifnsfcorp, 9)
train!(artifnsflda, iter=150, tol=0.0, chkelbo=15)

# training...

showtopics(artifnsflda, cols=9)



########################################
#                                      #
# Filtered Latent Dirichlet Allocation #
#                                      #
########################################

srand(1)

nsfflda = fLDA(nsfcorp, 9)
train!(nsfflda, iter=150, tol=0.0)

# training...

showtopics(nsfflda, cols=9)



###################################
#                                 #
# Filtered Correlated Topic Model #
#                                 #
###################################

srand(1)

nsffctm = fCTM(nsfcorp, 9)
train!(nsffctm, iter=150, tol=0.0)

# training...

showtopics(nsffctm, 20, cols=9)

nsffctm.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
nsffctm.sigma[4,8] # 9.532
nsffctm.sigma[3,6] # 7.362
nsffctm.sigma[2,9] # 4.531

# Top 3 negative entries, sorted in ascending order:
nsffctm.sigma[7,9] # -14.627
nsffctm.sigma[1,8] # -11.775
nsffctm.sigma[3,8] # -12.464



#######################
#                     #
# Dynamic Topic Model #
#                     #
#######################

import Distributions.sample

srand(1)

maccorp = readcorp(:mac)

maccorp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, maccorp.docs), 400, replace=false) for y in 1984:2005]...)

fixcorp!(maccorp, abr=100, len=10) # Remove words which appear < 100 times and documents of length < 10.

basemodel = LDA(maccorp, 9)
train!(basemodel, iter=150, chkelbo=151)

# training...

macdtm = DTM(maccorp, 9, 200, basemodel)
train!(macdtm, iter=10) # This will likely take several hours on a personal computer.
                        # Convergence for all other models is worst-case quadratic,
                        # while DTM convergence is linear or at best super-linear.
# training...

showtopics(model, topics=3, cols=6)

showtopics(model, times=11, cols=9)



#############################################
#                                           #
# Collaborative Topic Poisson Factorization #
#                                           #
#############################################

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

sum([isempty(doc.readers) for doc in citeucorp])

srand(1)

citeuctpf = gpuCTPF(citeucorp, 30) # Note: If no 'basemodel' is entered then parameters will be initialized at random.
train!(citeuctpf, iter=20, chkelbo=21)

# training...

acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)[1]
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.908

srand(1)

basemodel = gpuLDA(citeucorp, 30)
train!(basemodel, iter=100, chkelbo=101)

# training...

citeuctpf = gpuCTPF(citeucorp, 30, basemodel)
train!(citeuctpf, iter=20, chkelbo=21)

acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)[1]
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.920

testukeys[1] # = 216
acc[1] # = 0.945

showdrecs(citeuctpf, 1, 272, cols=1)

showurecs(citeuctpf, 216, 1382)

showlibs(citeuctpf, 1741)

showurecs(citeuctpf, 1741, 20)



####################
#                  #
# GPU Acceleration #
#                  #
####################

nsfcorp = readcorp(:nsf)

nsflda = LDA(nsfcorp, 16)
@time @gpu train!(nsflda, iter=150, chkelbo=151) # Let's time it as well to get an exact benchmark.

# training...

citeucorp = readcorp(:citeu)

citeuctm = CTM(citeucorp, 7)
@gpu 4250 train!(citeuctm, iter=150, chkelbo=25) # batchsize = 4250 documents.

# training...

