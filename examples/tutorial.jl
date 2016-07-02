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

artificialcorp = gencorp(nsflda, 5000, 1e-5) # The third argument governs the amount of Laplace smoothing (defaults to 0.0).

artificiallda = LDA(artificialcorp, 9)
train!(artificiallda, iter=150, tol=0.0, chkelbo=15)

# training...

showtopics(artificiallda, cols=9)



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

model.sigma

# Top 3 off-diagonal positive entries, sorted in descending order:
model.sigma[4,8] # 9.315
model.sigma[3,6] # 6.522
model.sigma[2,9] # 5.148

# Top 3 negative entries, sorted in ascending order:
model.sigma[7,9] # -13.212
model.sigma[1,8] # -13.134
model.sigma[3,8] # -11.429



#######################
#                     #
# Dynamic Topic Model #
#                     #
#######################

srand(1)

maccorp = readcorp(:mac)

maccorp.docs = vcat([sample(filter(doc -> round(doc.stamp / 100) == y, maccorp.docs), 400, replace=false) for y in 1984:2005]...)

fixcorp!(maccorp, b=100, len=10) # Remove words which appear < 100 times and documents of length < 10.

pmodel = LDA(corp, 9)
train!(pmodel, iter=150, chkelbo=151)

# training...

macdtm = DTM(maccorp, 9, 200, pmodel)
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

sum([isempty(doc.readers) for doc in corp]) # = 158

citeuctpf = CTPF(citeucorp, 30) # Note: 'pmodel' defaults to a 100 iteration LDA model.
train!(citeuctpf, iter=5)       # Instantiation and training will likely take 30 - 40 minutes on a personal computer.
                                # All optimizations in CTPF are analytic, often allowing for very fast convergence.
# training...

acc = Float64[]
for (d, u) in enumerate(testukeys)
    rank = findin(citeuctpf.drecs[d], u)[1]
    nrlen = length(citeuctpf.drecs[d])
    push!(acc, (nrlen - rank) / (nrlen - 1))
end

@show mean(acc) # mean(acc) = 0.913

testukeys[1] # = 216
acc[1] # = 0.973

showdrecs(model, 1, 152, cols=1)

showurecs(model, 216, 426)

showlibs(citeuctpf, 216)

showurecs(citeuctpf, 216, 10)

