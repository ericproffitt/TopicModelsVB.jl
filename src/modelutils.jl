#############################
#							#
# General Display Functions #
#							#
#############################

showdocs(model::TopicModel, ds::Vector{Int}) = showdocs(model.corp, ds)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, ds::UnitRange{Int}) = showdocs(model.corp, collect(ds))
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)

getlex(model::TopicModel) = sort(collect(values(model.corp.lex)))
getusers(model::TopicModel) = sort(collect(values(model.corp.users)))

Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::fLDA) = print(io, "Filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::CTM) = print(io, "Correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::fCTM) = print(io, "Filtered correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::DTM) = print(io, "Dynamic topic model with $(model.K) topics and ∆ = $(model.delta).")
Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")

Base.show(io::IO, model::memLDA) = print(io, "Low memory latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::memfLDA) = print(io, "Low memory filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::memCTM) = print(io, "Low memory correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::memfCTM) = print(io, "Low memory filtered correlated topic model with $(model.K) topics.")

Base.show(io::IO, model::gpuLDA) = print(io, "GPU accelerated latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTPF) = print(io, "GPU accelerated collaborative topic Poisson factorization model with $(model.K) topics.")



#############################################################################################
#						   																	#
# Function for Aligning Auxiliary Data with Primary Data Coupled with Primary Data Checking #
#						   																	#
#############################################################################################

function fixmodel!(model::LDA)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])	
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)

	model.Elogtheta = [digamma(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in 1:model.M]
	nothing
end

function fixmodel!(model::fLDA)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert isequal(length(model.alpha), model.K)	
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert (0 <= model.eta <= 1)	
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])	
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)

	model.Elogtheta = [digamma(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in 1:model.M]
	nothing
end

function fixmodel!(model::CTM)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert all(isfinite(model.mu))	
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])	
	@assert all(isfinite(model.lzeta))	
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)

	model.invsigma = inv(model.sigma)
	nothing
end

function fixmodel!(model::fCTM)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert (0 <= model.eta <= 1)	
	@assert all(isfinite(model.mu))
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)	
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])	
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])	
	@assert all(isfinite(model.lzeta))	
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)

	model.invsigma = inv(model.sigma)
	nothing
end

function fixmodel!(model::DTM)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert !isnegative(model.T)
	@assert isequal(vcat(model.S...), sortperm([doc.stamp for doc in model.corp]))	
	@assert isfinite(model.sigmasq)
	@assert ispositive(model.sigmasq)	
	@assert isequal(length(model.alpha), model.T)
	@assert all(Bool[isequal(length(model.alpha[t]), model.K) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.alpha[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.alpha[t])) for t in 1:model.T])	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isequal(size(model.m0), (model.K, model.V))
	@assert all(isfinite(model.m0))	
	@assert isequal(size(model.v0), (model.K, model.V))
	@assert all(isfinite(model.v0))
	@assert all(ispositive(model.v0))	
	@assert isequal(length(model.m), model.T)
	@assert all(Bool[isequal(size(model.m[t]), (model.K, model.V)) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.m[t])) for t in 1:model.T])
	@assert isequal(length(model.v), model.T)
	@assert all(Bool[isequal(size(model.v[t]), (model.K, model.V)) for t in 1:model.T])	
	@assert all(Bool[all(isfinite(model.v[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.v[t])) for t in 1:model.T])	
	@assert isequal(length(model.bsq), model.T)
	@assert all(isfinite(model.bsq))
	@assert all(ispositive(model.bsq))	
	@assert all(Bool[all(isfinite(model.betahat[t])) for t in 1:model.T])
	@assert isequal(size(model.mbeta0), (model.K, model.V))	
	@assert all(isfinite(model.mbeta0))	
	@assert isequal(size(model.vbeta0), (model.K, model.V))
	@assert all(isfinite(model.vbeta0))
	@assert all(ispositive(model.vbeta0))
	@assert isequal(length(model.mbeta), model.T)
	@assert all(Bool[isequal(size(model.mbeta[t]), (model.K, model.V)) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.mbeta[t])) for t in 1:model.T])
	@assert isequal(length(model.vbeta), model.T)
	@assert all(Bool[isequal(size(model.vbeta[t]), (model.K, model.V)) for t in 1:model.T])	
	@assert all(Bool[all(isfinite(model.vbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.vbeta[t])) for t in 1:model.T])
	@assert all(isfinite(model.lzeta))	
	@assert isfinite(model.delta)
	@assert ispositive(model.delta)	
	@assert isfinite(model.elbo)

	model.Eexpbeta = [exp(model.mbeta[t] + 0.5 * model.vbeta[t]) for t in 1:model.T]
	model.a = [maximum(model.Eexpbeta[t]) for t in 1:model.T]
	model.rEexpbeta = [exp(model.mbeta[t] + 0.5 * model.vbeta[t] - model.a[t]) for t in 1:model.T]
	nothing
end

function fixmodel!(model::CTPF)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])
	@assert isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])
	@assert isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])
	@assert ispositive(model.a)
	@assert ispositive(model.b)
	@assert ispositive(model.c)
	@assert ispositive(model.d)
	@assert ispositive(model.e)
	@assert ispositive(model.f)
	@assert ispositive(model.g)
	@assert ispositive(model.h)	
	@assert isequal(size(model.alef), (model.K, model.V))
	@assert all(isfinite(model.alef))
	@assert all(ispositive(model.alef))
	@assert isequal(length(model.bet), model.K)
	@assert all(isfinite(model.bet))
	@assert all(ispositive(model.bet))
	@assert isequal(length(model.gimel), model.M)
	@assert all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gimel[d])) for d in 1:model.M])
	@assert isequal(length(model.dalet), model.K)
	@assert all(isfinite(model.dalet))
	@assert all(ispositive(model.dalet))
	@assert isequal(size(model.he), (model.K, model.U))	
	@assert all(isfinite(model.he))
	@assert all(ispositive(model.he))
	@assert isequal(length(model.vav), model.K)
	@assert all(isfinite(model.vav))
	@assert all(ispositive(model.vav))
	@assert isequal(length(model.zayin), model.M)
	@assert all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.zayin[d])) for d in 1:model.M])
	@assert isequal(length(model.het), model.K)
	@assert all(isfinite(model.het))
	@assert all(ispositive(model.het))
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert isequal(length(model.xi), model.M)
	@assert all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)
	nothing	
end

function fixmodel!(model::memLDA)
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.betamem), (model.K, model.V))
	@assert isequal(model.betamem, zeros(model.K, model.V))
	@assert isequal(length(model.gamma), model.K)
	@assert all(isfinite(model.gamma))
	@assert all(ispositive(model.gamma))
	@assert isequal(size(model.phi), (model.K, model.N[1])) # what if corpus is empty?
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	@assert isequal(model.elbomem, 0.0)

	model.Elogtheta = fill(digamma(model.gamma) - digamma(sum(model.gamma)), model.M)
	nothing
end

function fixmodel!(model::memfLDA)
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert isequal(length(model.alpha), model.K)	
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert (0 <= model.eta <= 1)	
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.betamem), (model.K, model.V))
	@assert isequal(model.betamem, zeros(model.K, model.V))
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert isequal(length(model.kappamem), model.V)
	@assert isequal(model.kappamem, zeros(model.V))	
	@assert isequal(length(model.gamma), model.K)
	@assert all(isfinite(model.gamma))
	@assert all(ispositive(model.gamma))
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])	
	@assert isequal(size(model.phi), (model.K, model.N[1])) # what if corpus is empty?
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	@assert isequal(model.elbomem, 0.0)

	model.Elogtheta = fill(digamma(model.gamma) - digamma(sum(model.gamma)), model.M)
	nothing
end

function fixmodel!(model::memCTM)
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert all(isfinite(model.mu))	
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.betamem), (model.K, model.V))
	@assert isequal(model.betamem, zeros(model.K, model.V))
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])	
	@assert isfinite(model.lzeta)	
	@assert isequal(size(model.phi), (model.K, model.N[1])) # what if corpus is empty?
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	@assert isequal(model.elbomem, 0.0)

	model.invsigma = inv(model.sigma)
	nothing
end

function fixmodel!(model::memfCTM)
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert (0 <= model.eta <= 1)	
	@assert all(isfinite(model.mu))
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.betamem), (model.K, model.V))
	@assert isequal(model.betamem, zeros(model.K, model.V))
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert isequal(length(model.kappamem), model.V)
	@assert isequal(model.kappamem, zeros(model.V))
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])	
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])	
	@assert isfinite(model.lzeta)
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])
	@assert isequal(size(model.phi), (model.K, model.N[1])) # what if corpus is empty?
	@assert isprobvec(model.phi, 1)	
	@assert isfinite(model.elbo)
	@assert isequal(model.elbomem, 0.0)

	model.invsigma = inv(model.sigma)
	nothing
end

function fixmodel!(model::gpuLDA)
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])	
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)

	model.Elogtheta = [digamma(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in 1:model.M]
	model.sumElogtheta = sum(model.Elogtheta)
	
	model.device, model.context, model.queue = OpenCL.create_compute_context()		

	terms = vcat([doc.terms for doc in model.corp]...) - 1
	counts = vcat([doc.counts for doc in model.corp]...)
	words = sortperm(terms) - 1

	Npsums = zeros(Int, model.M + 1)
	for d in 1:model.M
		Npsums[d+1] = Npsums[d] + model.N[d]
	end
		
	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	Jpsums = zeros(Int, model.V + 1)
	for j in 1:model.V
		Jpsums[j+1] = Jpsums[j] + J[j]
	end

	model.Npsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Npsums)
	model.Jpsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Jpsums)
	model.terms = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)	
	model.counts = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)		
	model.words = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=words)		

	betaprog = OpenCL.Program(model.context, source=LDAbetacpp) |> OpenCL.build!
	betanormprog = OpenCL.Program(model.context, source=LDAbetanormcpp) |> OpenCL.build!
	gammaprog = OpenCL.Program(model.context, source=LDAgammacpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=LDAphicpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=LDAphinormcpp) |> OpenCL.build!
	Elogthetaprog = OpenCL.Program(model.context, source=LDAElogthetacpp) |> OpenCL.build!
	sumElogthetaprog = OpenCL.Program(model.context, source=LDAsumElogthetacpp) |> OpenCL.build!

	model.betakern = OpenCL.Kernel(betaprog, "updateBeta")
	model.betanormkern = OpenCL.Kernel(betanormprog, "normalizeBeta")
	model.gammakern = OpenCL.Kernel(gammaprog, "updateGamma")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
	model.Elogthetakern = OpenCL.Kernel(Elogthetaprog, "updateElogtheta")
	model.sumElogthetakern = OpenCL.Kernel(sumElogthetaprog, "updatesumElogtheta")
	updateBuf!(model)	
	nothing
end

function fixmodel!(model::gpuCTPF)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])
	@assert isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])
	@assert isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])
	@assert ispositive(model.a)
	@assert ispositive(model.b)
	@assert ispositive(model.c)
	@assert ispositive(model.d)
	@assert ispositive(model.e)
	@assert ispositive(model.f)
	@assert ispositive(model.g)
	@assert ispositive(model.h)	
	@assert isequal(size(model.alef), (model.K, model.V))
	@assert all(isfinite(model.alef))
	@assert all(ispositive(model.alef))
	@assert isequal(length(model.bet), model.K)
	@assert all(isfinite(model.bet))
	@assert all(ispositive(model.bet))
	@assert isequal(length(model.gimel), model.M)
	@assert all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gimel[d])) for d in 1:model.M])
	@assert isequal(length(model.dalet), model.K)
	@assert all(isfinite(model.dalet))
	@assert all(ispositive(model.dalet))
	@assert isequal(size(model.he), (model.K, model.U))	
	@assert all(isfinite(model.he))
	@assert all(ispositive(model.he))
	@assert isequal(length(model.vav), model.K)
	@assert all(isfinite(model.vav))
	@assert all(ispositive(model.vav))
	@assert isequal(length(model.zayin), model.M)
	@assert all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.zayin[d])) for d in 1:model.M])
	@assert isequal(length(model.het), model.K)
	@assert all(isfinite(model.het))
	@assert all(ispositive(model.het))
	@assert isequal(length(model.phi), model.M)
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert isequal(length(model.xi), model.M)
	@assert all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in 1:model.M])	
	@assert isfinite(model.elbo)
		
	model.device, model.context, model.queue = OpenCL.create_compute_context()		

	terms = vcat([doc.terms for doc in model.corp]...) - 1
	counts = vcat([doc.counts for doc in model.corp]...)
	words = sortperm(terms) - 1

	readers = vcat([doc.readers for doc in model.corp]...) - 1
	ratings = vcat([doc.ratings for doc in model.corp]...)
	views = sortperm(readers) - 1

	Npsums = zeros(Int, model.M + 1)
	Rpsums = zeros(Int, model.M + 1)
	for d in 1:model.M
		Npsums[d+1] = Npsums[d] + model.N[d]
		Rpsums[d+1] = Rpsums[d] + model.R[d]
	end

	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	Jpsums = zeros(Int, model.V + 1)
	for j in 1:model.V
		Jpsums[j+1] = Jpsums[j] + J[j]
	end

	Y = zeros(Int, model.U)
	for r in readers
		Y[r+1] += 1
	end

	Ypsums = zeros(Int, model.U + 1)
	for u in 1:model.U
		Ypsums[u+1] = Ypsums[u] + Y[u]
	end

	model.Npsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Npsums)
	model.Jpsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Jpsums)
	model.terms = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.counts = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)
	model.words = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=words)

	model.Rpsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Rpsums)
	model.Ypsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Ypsums)
	model.readers = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=readers)
	model.ratings = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=ratings)
	model.views = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=views)

	alefprog = OpenCL.Program(model.context, source=CTPFalefcpp) |> OpenCL.build!
	betprog = OpenCL.Program(model.context, source=CTPFbetcpp) |> OpenCL.build!
	gimelprog = OpenCL.Program(model.context, source=CTPFgimelcpp) |> OpenCL.build!
	daletprog = OpenCL.Program(model.context, source=CTPFdaletcpp) |> OpenCL.build!
	heprog = OpenCL.Program(model.context, source=CTPFhecpp) |> OpenCL.build!
	vavprog = OpenCL.Program(model.context, source=CTPFvavcpp) |> OpenCL.build!
	zayinprog = OpenCL.Program(model.context, source=CTPFzayincpp) |> OpenCL.build!
	hetprog = OpenCL.Program(model.context, source=CTPFhetcpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=CTPFphicpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=CTPFphinormcpp) |> OpenCL.build!
	xiprog = OpenCL.Program(model.context, source=CTPFxicpp) |> OpenCL.build!
	xinormprog = OpenCL.Program(model.context, source=CTPFxinormcpp) |> OpenCL.build!

	model.alefkern = OpenCL.Kernel(alefprog, "updateAlef")
	model.betkern = OpenCL.Kernel(betprog, "updateBet")
	model.gimelkern = OpenCL.Kernel(gimelprog, "updateGimel")
	model.daletkern = OpenCL.Kernel(daletprog, "updateDalet")
	model.hekern = OpenCL.Kernel(heprog, "updateHe")
	model.vavkern = OpenCL.Kernel(vavprog, "updateVav")
	model.zayinkern = OpenCL.Kernel(zayinprog, "updateZayin")
	model.hetkern = OpenCL.Kernel(hetprog, "updateHet")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
	model.xikern = OpenCL.Kernel(xiprog, "updateXi")
	model.xinormkern = OpenCL.Kernel(xinormprog, "normalizeXi")
	updateBuf!(model)
	nothing
end



##################################################
#												 #
# Function for Updating the Evidence Lower Bound #
#												 #
##################################################

function checkELBO!(model::TopicModel, k::Int, chkelbo::Integer, tol::Real)
	converged = false
	if k % chkelbo == 0
		∆elbo = -(model.elbo - updateELBO!(model))
		println(k, " ∆elbo: ", round(∆elbo, 3))
		if abs(∆elbo) < tol
			converged = true
		end
	end

	return converged
end

function checkELBO!(model::GPUTopicModel, k::Int, chkelbo::Integer, tol::Real)
	converged = false
	if k % chkelbo == 0
		updateHost!(model)
		∆elbo = -(model.elbo - updateELBO!(model))
		println(k, " ∆elbo: ", round(∆elbo, 3))
		if abs(∆elbo) < tol
			converged = true
		end
	end

	return converged
end



##############################################################
#															 #
# Host-to-Buffer and Buffer-to-Host Functions for GPU Models #
#															 #
##############################################################

function updateBuf!(model::gpuLDA)
	model.alphabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alpha)
	model.betabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.gammabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gamma...))
	model.phibuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.phi...))
	model.Elogthetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.Elogtheta...))
	model.sumElogthetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.sumElogtheta)
end

function updateBuf!(model::gpuCTPF)
	model.alefbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alef)
	model.betbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.bet)
	model.gimelbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gimel...))
	model.daletbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.dalet)
	model.hebuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.he)
	model.vavbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.vav)
	model.zayinbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.zayin...))
	model.hetbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.het)
	model.phibuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.phi...))
	model.xibuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.xi...))
end

function updateHost!(model::gpuLDA)
	model.alpha = OpenCL.read(model.queue, model.alphabuf)
	model.beta = reshape(OpenCL.read(model.queue, model.betabuf), model.K, model.V)
	hostgamma = reshape(OpenCL.read(model.queue, model.gammabuf), model.K, model.M)
	@bumper model.gamma = [hostgamma[:,d] for d in 1:model.M]
	hostphi = reshape(OpenCL.read(model.queue, model.phibuf), model.K, sum(model.N))
	Npsums = OpenCL.read(model.queue, model.Npsums)
	model.phi = [hostphi[:,Npsums[d]+1:Npsums[d+1]] for d in 1:model.M]
	hostElogtheta = reshape(OpenCL.read(model.queue, model.Elogthetabuf), model.K, model.M)
	model.Elogtheta = [hostElogtheta[:,d] for d in 1:model.M]
	model.sumElogtheta = OpenCL.read(model.queue, model.sumElogthetabuf)
end

function updateHost!(model::gpuCTPF)
	model.alef = reshape(OpenCL.read(model.queue, model.alefbuf), model.K, model.V)
	model.bet = OpenCL.read(model.queue, model.betbuf)
	hostgimel = reshape(OpenCL.read(model.queue, model.gimelbuf), model.K, model.M)
	model.gimel = [hostgimel[:,d] for d in 1:model.M]
	model.dalet = OpenCL.read(model.queue, model.daletbuf)
	model.he = reshape(OpenCL.read(model.queue, model.hebuf), model.K, model.U)
	model.vav = OpenCL.read(model.queue, model.vavbuf)
	hostzayin = reshape(OpenCL.read(model.queue, model.zayinbuf), model.K, model.M)
	model.zayin = [hostzayin[:,d] for d in 1:model.M]
	model.het = OpenCL.read(model.queue, model.hetbuf)
	Npsums = OpenCL.read(model.queue, model.Npsums)	
	hostphi = reshape(OpenCL.read(model.queue, model.phibuf), model.K, sum(model.N))
	model.phi = [hostphi[:,Npsums[d]+1:Npsums[d+1]] for d in 1:model.M]
	Rpsums = OpenCL.read(model.queue, model.Rpsums)
	hostxi = reshape(OpenCL.read(model.queue, model.xibuf), 2model.K, sum(model.R))
	model.xi = [hostxi[:,Rpsums[d]+1:Rpsums[d+1]] for d in 1:model.M]
end



#############################################################
#															#
# Functions for Generating Artificial Documents and Corpora #
#															#
#############################################################

function gendoc(model::Union{LDA, memLDA, gpuLDA}, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.beta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gendoc(model::Union{fLDA, memfLDA}, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.fbeta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gendoc(model::Union{CTM, memCTM}, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp(theta) / sum(exp(theta))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.beta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gendoc(model::Union{fCTM, memfCTM}, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp(theta) / sum(exp(theta))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.fbeta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gencorp(model::BaseTopicModel, corpsize::Int, a::Real=0.0)
	@assert ispositive(corpsize)
	@assert !isnegative(a)
	
	corp = Corpus(lex=model.corp.lex, users=model.corp.users)
	corp.docs = [gendoc(model, a) for d in 1:corpsize]

	return corp
end



###########################
#						  #
# Topic Display Functions #
#						  #
###########################

function showtopics(model::TopicModel, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), cols::Int=4)
	@assert checkbounds(Bool, model.V, N)
	@assert checkbounds(Bool, model.K, topics)
	@assert ispositive(cols)
	isa(topics, Vector) || (topics = [topics])
	cols = min(cols, length(topics))

	lex = model.corp.lex
	maxjspacings = [maximum([length(lex[j]) for j in topic[1:N]]) for topic in model.topics]

	for block in partition(topics, cols)
		for j in 0:N
			for (k, i) in enumerate(block)
				if j == 0
					jspacing = max(4, maxjspacings[i] - length("$i") - 2)
					k == cols ? yellow("topic $i") : yellow("topic $i" * " "^jspacing)
				else
					jspacing = max(6 + length("$i"), maxjspacings[i]) - length(lex[model.topics[i][j]]) + 4
					k == cols ? print(lex[model.topics[i][j]]) : print(lex[model.topics[i][j]] * " "^jspacing)
				end
			end
			println()
		end
		println()
	end
end

function showtopics(model::DTM, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), times::Union{Int, Vector{Int}}=collect(1:model.T), cols::Int=4)
	@assert checkbounds(Bool, model.V, N)
	@assert checkbounds(Bool, model.K, topics)
	@assert checkbounds(Bool, model.T, times)
	@assert ispositive(cols)
	isa(times, Vector) || (times = [times])
	
	corp, lex = model.corp, model.corp.lex

	if length(topics) > 1
		container = LDA(Corpus(), model.K)
		for t in times
			container.corp = corp
			container.topics = model.topics[t][topics]
			container.V = model.V
			@juliadots "Time: $t\n"
			@juliadots "Span: $(corp[model.S[t][1]].stamp) - $(corp[model.S[t][end]].stamp)\n"
			showtopics(container, N, topics=topics, cols=cols)
		end
	
	else
		cols = min(cols, length(times))
		@juliadots "Topic: $(topics[1])\n"
		maxjspacings = [maximum([length(lex[j]) for j in time[topics[1]][1:N]]) for time in model.topics]

		for block in partition(times, cols)
			for j in 0:N
				for (s, t) in enumerate(block)
					if j == 0
						jspacing = max(4, maxjspacings[t] - length("$t") - 1)
						s == cols ? yellow("time $t") : yellow("time $t" * " "^jspacing)
					else
						jspacing = max(5 + length("$t"), maxjspacings[t]) - length(lex[model.topics[t][topics[1]][j]]) + 4
						s == cols ? print(lex[model.topics[t][topics[1]][j]]) : print(lex[model.topics[t][topics[1]][j]] * " "^jspacing)
					end
				end
				println()
			end
			println()
		end
	end
end

function showlibs(model::Union{CTPF, gpuCTPF}, users::Vector{Int})
	@assert checkbounds(Bool, model.U, users)
	
	for u in users
		@juliadots "User: $u\n"
		try if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		catch @juliadots model.corp.users[u] * "\n"
		end
		
		for d in model.libs[u]
			yellow(" • ")
			isempty(model.corp[d].title) ? bold("Doc: $d\n") : bold("$(model.corp[d].title)\n")
		end
		println()
	end
end

showlibs(model::Union{CTPF, gpuCTPF}, user::Int) = showlibs(model, [user])

function showdrecs(model::Union{CTPF, gpuCTPF}, docs::Union{Int, Vector{Int}}, U::Int=min(16, model.U); cols::Int=4)
	@assert checkbounds(Bool, model.M, docs)	
	@assert checkbounds(Bool, model.U, U)
	@assert ispositive(cols)
	isa(docs, Vector) || (docs = [docs])
	corp, drecs, users = model.corp, model.drecs, model.corp.users

	for d in docs
		@juliadots "Doc: $d\n"
		if !isempty(corp[d].title)
			@juliadots corp[d].title * "\n"
		end

		usercols = partition(drecs[d][1:U], Int(ceil(U / cols)))
		rankcols = partition(1:U, Int(ceil(U / cols)))

		for i in 1:length(usercols[1])
			for j in 1:length(usercols)
				try
				uspacing = maximum([length(users[u]) for u in usercols[j]]) - length(users[usercols[j][i]]) + 4
				rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
				yellow(string(rankcols[j][i]) * ". " * " "^rspacing)
				j == length(usercols) ? print(users[usercols[j][i]]) : print(users[usercols[j][i]] * " "^uspacing)
				end
			end
			println()
		end
		println()
	end
end

function showurecs(model::Union{CTPF, gpuCTPF}, users::Union{Int, Vector{Int}}, M::Int=min(10, model.M); cols::Int=1)
	@assert checkbounds(Bool, model.U, users)
	@assert checkbounds(Bool, model.M, M)
	@assert ispositive(cols)
	isa(users, Vector) || (users = [users])
	corp, urecs, docs = model.corp, model.urecs, model.corp.docs

	for u in users
		@juliadots "User: $u\n"
		try if corp.users[u][1:5] != "#user"
				@juliadots corp.users[u] * "\n"
			end
		catch @juliadots corp.users[u] * "\n"
		end

		docucols = partition(urecs[u][1:M], Int(ceil(M / cols)))
		rankcols = partition(1:M, Int(ceil(M / cols)))

		for i in 1:length(docucols[1])
			for j in 1:length(docucols)
				try
				!isempty(corp[docucols[j][i]].title) ? title = corp[docucols[j][i]].title : title = "doc $(docucols[j][i])"
				dspacing = maximum([max(4 + length("$(docucols[j][i])"), length(docs[d].title)) for d in docucols[j]]) - length(title) + 4
				rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
				yellow(string(rankcols[j][i]) * ". " * " "^rspacing)
				j == length(docucols) ? bold(title) : bold(title * " "^dspacing)
				end
			end
			println()
		end
		println()
	end
end

