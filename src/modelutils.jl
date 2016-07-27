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

	model.alefbet = vec(sum(model.alef ./ model.bet, 2))
	model.hevav = vec(sum(model.he ./ model.vav, 2))
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
	model.SUMElogtheta = sum(model.Elogtheta)
	
	model.device, model.context, model.queue = OpenCL.create_compute_context()		

	terms = vcat([doc.terms for doc in model.corp]...)
	words = collect(0:sum(model.N)-1)[sortperm(terms)]
	J = zeros(Int, model.V)
	for j in terms
		J[j] += 1
	end

	Npsums = Int[0]
	for d in 1:model.M
		push!(Npsums, Npsums[d] + model.N[d])
	end
		
	model.terms = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=terms - 1)	
	model.counts = OpenCL.Buffer(Float32, model.context, (:r, :copy), hostbuf=map(Float32, vcat([doc.counts for doc in model.corp]...)))		
	model.Npsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=Npsums)
	model.Jpsums = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=[sum(J[1:j]) for j in 0:model.V])
	model.words = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=words)		

	betaprog = OpenCL.Program(model.context, source=LDAbetacpp) |> OpenCL.build!
	betanormprog = OpenCL.Program(model.context, source=LDAbetanormcpp) |> OpenCL.build!
	gammaprog = OpenCL.Program(model.context, source=LDAgammacpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=LDAphicpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=LDAphinormcpp) |> OpenCL.build!
	Elogthetaprog = OpenCL.Program(model.context, source=LDAElogthetacpp) |> OpenCL.build!
	SUMElogthetaprog = OpenCL.Program(model.context, source=LDASUMElogthetacpp) |> OpenCL.build!

	model.betakern = OpenCL.Kernel(betaprog, "updateBeta")
	model.betanormkern = OpenCL.Kernel(betanormprog, "normalizeBeta")	
	model.gammakern = OpenCL.Kernel(gammaprog, "updateGamma")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
	model.Elogthetakern = OpenCL.Kernel(Elogthetaprog, "updateElogtheta")
	model.SUMElogthetakern = OpenCL.Kernel(SUMElogthetaprog, "updateSUMElogtheta")
	
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
	@buf model.alpha
	@buf model.beta
	@buf model.gamma
	@buf model.phi
	@buf model.Elogtheta
	@buf model.SUMElogtheta
end

function updateHost!(model::gpuLDA)
	@host model.alphabuf
	@host model.betabuf
	@host model.gammabuf
	@host model.phibuf
	@host model.Elogthetabuf
	@host model.SUMElogthetabuf
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

function showlibs(model::CTPF, users::Vector{Int})
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

showlibs(model::CTPF, user::Int) = showlibs(model, [user])

function showdrecs(model::CTPF, docs::Union{Int, Vector{Int}}, U::Int=min(16, model.U); cols::Int=4)
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

function showurecs(model::CTPF, users::Union{Int, Vector{Int}}, M::Int=min(10, model.M); cols::Int=1)
	@assert checkbounds(Bool, model.U, users)
	@assert checkbounds(Bool, model.M, M)
	@assert ispositive(cols)
	checkmodel(model)
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

