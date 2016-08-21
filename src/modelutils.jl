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

Base.show(io::IO, model::gpuLDA) = print(io, "GPU accelerated latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTM) = print(io, "GPU accelerated correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTPF) = print(io, "GPU accelerated collaborative topic Poisson factorization model with $(model.K) topics.")



##############################################################
#															 #
# Host-to-Buffer and Buffer-to-Host Functions for GPU Models #
#															 #
##############################################################

function updateBuf!(model::gpuLDA, b::Int)
	b = b % model.B + 1

	@buf b model.Npsums
	@buf b model.Jpsums
	@buf b model.terms
	@buf b model.counts
	@buf b model.words

	@buf b model.phi
	@buf b model.Elogtheta
end

function updateBuf!(model::gpuCTM, b::Int)
	b = b % model.B + 1

	@buf b model.C
	@buf b model.Npsums
	@buf b model.Jpsums
	@buf b model.terms
	@buf b model.counts
	@buf b model.words

	@buf b model.newtontemp
	@buf b model.newtongrad
	@buf b model.newtoninvhess

	@buf b model.phi
end

function updateBuf!(model::gpuCTPF, b::Int)
	b = b % model.B + 1

	@buf b model.Npsums
	@buf b model.Jpsums
	@buf b model.Rpsums
	@buf b model.Ypsums
	@buf b model.terms
	@buf b model.counts
	@buf b model.words
	@buf b model.readers
	@buf b model.ratings
	@buf b model.views

	@buf b model.phi
	@buf b model.xi
end

function updateHost!(model::gpuLDA, b::Int)
	@host model.alphabuf
	@host model.betabuf
	@host model.gammabuf
	@host b model.phibuf
	@host b model.Elogthetabuf
	@host b model.Elogthetasumbuf
end

function updateHost!(model::gpuCTM, b::Int)
	@host model.mubuf
	@host model.sigmabuf
	@host model.invsigmabuf
	@host model.betabuf
	@host model.lambdabuf
	@host model.vsqbuf
	@host model.lzetabuf
	@host b model.phibuf
end

function updateHost!(model::gpuCTPF, b::Int)
	@host model.alefbuf
	@host model.betbuf
	@host model.gimelbuf
	@host model.daletbuf
	@host model.hebuf
	@host model.vavbuf
	@host model.zayinbuf
	@host model.hetbuf
	@host b model.phibuf
	@host b model.xibuf
end



######################################################################################################
#						   																			 #
# Function for Aligning Auxiliary Data with Primary Data Coupled with Optional Primary Data Checking #
#						   																			 #
######################################################################################################

function fixmodel!(model::LDA; check::Bool=true)
	if check
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
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	end

	model.Elogtheta = digamma(model.gamma[1]) - digamma(sum(model.gamma[1]))
	model.Elogthetasum = zeros(model.K)
	model.newbeta = zeros(model.K, model.V)
	model.newelbo = 0
	nothing
end

function fixmodel!(model::fLDA; check::Bool=true)
	if check
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
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert isequal(length(model.newkappa), model.V)
	@assert isequal(model.newkappa, zeros(model.V))	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])	
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	end

	model.Elogtheta = digamma(model.gamma[1]) - digamma(sum(model.gamma[1]))
	model.Elogthetasum = zeros(model.K)
	model.newbeta = zeros(model.K, model.V)
	model.newkappa = zeros(model.V)
	model.newelbo = 0
	nothing
end

function fixmodel!(model::CTM; check::Bool=true)
	if check
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
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])	
	@assert isfinite(model.lzeta)	
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	end

	model.invsigma = inv(model.sigma)
	model.newbeta = zeros(model.K, model.V)
	model.newelbo = 0
	nothing
end

function fixmodel!(model::fCTM; check::Bool=true)
	if check
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
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)	
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert isequal(length(model.newkappa), model.V)
	@assert isequal(model.newkappa, zeros(model.V))
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
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)	
	@assert isfinite(model.elbo)
	end

	model.invsigma = inv(model.sigma)
	model.newbeta = zeros(model.K, model.V)
	model.newkappa = zeros(model.V)
	model.newelbo = 0
	nothing
end

function fixmodel!(model::DTM; check::Bool=true)
	if check
	checkcorp(model.corp)
	@assert !isempty(model.corp)
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
	end

	model.Elogtheta = [digamma(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in 1:model.M]
	model.Eexpbeta = [exp(model.mbeta[t] + 0.5 * model.vbeta[t]) for t in 1:model.T]
	model.maxlEexpbeta = [maximum(model.Eexpbeta[t]) for t in 1:model.T]
	model.ovflEexpbeta = [exp(model.mbeta[t] + 0.5 * model.vbeta[t] - model.maxlEexpbeta[t]) for t in 1:model.T]
	nothing
end

function fixmodel!(model::CTPF; check::Bool=true)
	if check
	checkcorp(model.corp)
	@assert !isempty(model.corp)
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
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isequal(size(model.xi), (2model.K, model.R[1]))
	@assert isprobvec(model.xi, 1)
	@assert isfinite(model.elbo)
	end

	model.newalef = fill(model.a, model.K, model.V)
	model.newhe = fill(model.e, model.K, model.U)
	model.newelbo = 0
	nothing	
end

function fixmodel!(model::gpuLDA; check::Bool=true)
	if check
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
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
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])
	end

	model.Elogtheta = [digamma(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in model.batches[1]]
	model.Elogthetasum = zeros(model.K)
	model.newbeta = nothing
	
	model.terms = [vcat([doc.terms for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.counts = [vcat([doc.counts for doc in model.corp[batch]]...) for batch in model.batches]
	model.words = [sortperm(termvec) - 1 for termvec in model.terms]

	model.Npsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	for (b, batch) in enumerate(model.batches)
		for (n, d) in enumerate(batch)
			model.Npsums[b][n+1] = model.Npsums[b][n] + model.N[d]
		end
	end
		
	J = [zeros(Int, model.V) for _ in 1:model.B]
	for b in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for b in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	model.device, model.context, model.queue = OpenCL.create_compute_context()

	betaprog = OpenCL.Program(model.context, source=LDA_BETA_cpp) |> OpenCL.build!
	betanormprog = OpenCL.Program(model.context, source=LDA_BETA_NORM_cpp) |> OpenCL.build!
	newbetaprog = OpenCL.Program(model.context, source=LDA_NEWBETA_cpp) |> OpenCL.build!
	gammaprog = OpenCL.Program(model.context, source=LDA_GAMMA_cpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=LDA_PHI_cpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=LDA_PHI_NORM_cpp) |> OpenCL.build!
	Elogthetaprog = OpenCL.Program(model.context, source=LDA_ELOGTHETA_cpp) |> OpenCL.build!
	Elogthetasumprog = OpenCL.Program(model.context, source=LDA_ELOGTHETASUM_cpp) |> OpenCL.build!

	model.betakern = OpenCL.Kernel(betaprog, "updateBeta")
	model.betanormkern = OpenCL.Kernel(betanormprog, "normalizeBeta")
	model.newbetakern = OpenCL.Kernel(newbetaprog, "updateNewbeta")
	model.gammakern = OpenCL.Kernel(gammaprog, "updateGamma")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
	model.Elogthetakern = OpenCL.Kernel(Elogthetaprog, "updateElogtheta")
	model.Elogthetasumkern = OpenCL.Kernel(Elogthetasumprog, "updateElogthetasum")		

	@buf model.alpha
	@buf model.beta
	@buf model.gamma
	@buf model.Elogthetasum
	@buf model.newbeta
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end

function fixmodel!(model::gpuCTM; check::Bool=true)
	if check
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
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
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])
	end

	model.invsigma = inv(model.sigma)
	model.newbeta = nothing

	model.terms = [vcat([doc.terms for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.counts = [vcat([doc.counts for doc in model.corp[batch]]...) for batch in model.batches]
	model.words = [sortperm(termvec) - 1 for termvec in model.terms]

	model.Npsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	for (b, batch) in enumerate(model.batches)
		for (n, d) in enumerate(batch)
			model.Npsums[b][n+1] = model.Npsums[b][n] + model.N[d]
		end
	end
		
	J = [zeros(Int, model.V) for _ in 1:model.B]
	for b in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for b in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	model.device, model.context, model.queue = OpenCL.create_compute_context()

	muprog = OpenCL.Program(model.context, source=CTM_MU_cpp) |> OpenCL.build!
	betaprog = OpenCL.Program(model.context, source=CTM_BETA_cpp) |> OpenCL.build!
	betanormprog = OpenCL.Program(model.context, source=CTM_BETA_NORM_cpp) |> OpenCL.build!
	newbetaprog = OpenCL.Program(model.context, source=CTM_NEWBETA_cpp) |> OpenCL.build!
	lambdaprog = OpenCL.Program(model.context, source=CTM_LAMBDA_cpp) |> OpenCL.build!
	vsqprog = OpenCL.Program(model.context, source=CTM_VSQ_cpp) |> OpenCL.build!
	lzetaprog = OpenCL.Program(model.context, source=CTM_LZETA_cpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=CTM_PHI_cpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=CTM_PHI_NORM_cpp) |> OpenCL.build!

	model.mukern = OpenCL.Kernel(muprog, "updateMu")
	model.betakern = OpenCL.Kernel(betaprog, "updateBeta")
	model.betanormkern = OpenCL.Kernel(betanormprog, "normalizeBeta")
	model.newbetakern = OpenCL.Kernel(newbetaprog, "updateNewbeta")
	model.lambdakern = OpenCL.Kernel(lambdaprog, "updateLambda")
	model.vsqkern = OpenCL.Kernel(vsqprog, "updateVsq")
	model.lzetakern = OpenCL.Kernel(lzetaprog, "updateLzeta")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
		
	@buf model.mu
	@buf model.sigma
	@buf model.beta
	@buf model.lambda
	@buf model.vsq
	@buf model.lzeta
	@buf model.invsigma
	@buf model.newbeta
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end

function fixmodel!(model::gpuCTPF; check::Bool=true)
	if check
	checkcorp(model.corp)
	@assert !isempty(model.corp)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
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
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])
	@assert isequal(length(model.xi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in model.batches[1]])
	end

	model.newalef = nothing
	model.newhe = nothing
		
	model.terms = [vcat([doc.terms for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.counts = [vcat([doc.counts for doc in model.corp[batch]]...) for batch in model.batches]
	model.words = [sortperm(termvec) - 1 for termvec in model.terms]

	model.readers = [vcat([doc.readers for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.ratings = [vcat([doc.ratings for doc in model.corp[batch]]...) for batch in model.batches]
	model.views = [sortperm(readervec) - 1 for readervec in model.readers]

	model.Npsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	model.Rpsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	for (b, batch) in enumerate(model.batches)
		for (m, d) in enumerate(batch)
			model.Npsums[b][m+1] = model.Npsums[b][m] + model.N[d]
			model.Rpsums[b][m+1] = model.Rpsums[b][m] + model.R[d]
		end
	end
		
	J = [zeros(Int, model.V) for _ in 1:model.B]
	for b in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for b in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	Y = [zeros(Int, model.U) for _ in 1:model.B]
	for b in 1:model.B
		for r in model.readers[b]
			Y[b][r+1] += 1
		end
	end

	model.Ypsums = [zeros(Int, model.U + 1) for _ in 1:model.B]
	for b in 1:model.B
		for u in 1:model.U
			model.Ypsums[b][u+1] = model.Ypsums[b][u] + Y[b][u]
		end
	end

	model.device, model.context, model.queue = OpenCL.create_compute_context()		

	alefprog = OpenCL.Program(model.context, source=CTPF_ALEF_cpp) |> OpenCL.build!
	newalefprog = OpenCL.Program(model.context, source=CTPF_NEWALEF_cpp) |> OpenCL.build!
	betprog = OpenCL.Program(model.context, source=CTPF_BET_cpp) |> OpenCL.build!
	gimelprog = OpenCL.Program(model.context, source=CTPF_GIMEL_cpp) |> OpenCL.build!
	daletprog = OpenCL.Program(model.context, source=CTPF_DALET_cpp) |> OpenCL.build!
	heprog = OpenCL.Program(model.context, source=CTPF_HE_cpp) |> OpenCL.build!
	newheprog = OpenCL.Program(model.context, source=CTPF_NEWHE_cpp) |> OpenCL.build!
	vavprog = OpenCL.Program(model.context, source=CTPF_VAV_cpp) |> OpenCL.build!
	zayinprog = OpenCL.Program(model.context, source=CTPF_ZAYIN_cpp) |> OpenCL.build!
	hetprog = OpenCL.Program(model.context, source=CTPF_HET_cpp) |> OpenCL.build!
	phiprog = OpenCL.Program(model.context, source=CTPF_PHI_cpp) |> OpenCL.build!
	phinormprog = OpenCL.Program(model.context, source=CTPF_PHI_NORM_cpp) |> OpenCL.build!
	xiprog = OpenCL.Program(model.context, source=CTPF_XI_cpp) |> OpenCL.build!
	xinormprog = OpenCL.Program(model.context, source=CTPF_XI_NORM_cpp) |> OpenCL.build!

	model.alefkern = OpenCL.Kernel(alefprog, "updateAlef")
	model.newalefkern = OpenCL.Kernel(newalefprog, "updateNewalef")
	model.betkern = OpenCL.Kernel(betprog, "updateBet")
	model.gimelkern = OpenCL.Kernel(gimelprog, "updateGimel")
	model.daletkern = OpenCL.Kernel(daletprog, "updateDalet")
	model.hekern = OpenCL.Kernel(heprog, "updateHe")
	model.newhekern = OpenCL.Kernel(newheprog, "updateNewhe")
	model.vavkern = OpenCL.Kernel(vavprog, "updateVav")
	model.zayinkern = OpenCL.Kernel(zayinprog, "updateZayin")
	model.hetkern = OpenCL.Kernel(hetprog, "updateHet")
	model.phikern = OpenCL.Kernel(phiprog, "updatePhi")
	model.phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
	model.xikern = OpenCL.Kernel(xiprog, "updateXi")
	model.xinormkern = OpenCL.Kernel(xinormprog, "normalizeXi")
		
	@buf model.alef
	@buf model.bet
	@buf model.gimel
	@buf model.dalet
	@buf model.he
	@buf model.vav
	@buf model.zayin
	@buf model.het
	@buf model.newalef
	@buf model.newhe
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end



##################################################
#												 #
# Function for Updating the Evidence Lower Bound #
#												 #
##################################################

function checkELBO!(model::TopicModel, k::Int, chk::Bool, tol::Real)
	converged = false
	if chk
		∆elbo = -(model.elbo - updateELBO!(model))
		println(k, " ∆elbo: ", round(∆elbo, 3))
		if abs(∆elbo) < tol
			converged = true
		end
	end

	return converged
end



#############################################################
#															#
# Functions for Generating Artificial Documents and Corpora #
#															#
#############################################################

function gendoc(model::AbstractLDA, a::Real=0.0)
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

function gendoc(model::AbstractfLDA, a::Real=0.0)
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

function gendoc(model::AbstractCTM, a::Real=0.0)
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

function gendoc(model::AbstractfCTM, a::Real=0.0)
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

function gencorp(model::BaseTopicModel, corpsize::Integer, a::Real=0.0)
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

function showtopics{T<:Integer}(model::TopicModel, N::Integer=min(15, model.V); topics::Union{T, Vector{T}}=collect(1:model.K), cols::Integer=4)
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

function showtopics{T<:Integer, S<:Integer}(model::AbstractDTM, N::Integer=min(15, model.V); topics::Union{T, Vector{T}}=collect(1:model.K), times::Union{S, Vector{S}}=collect(1:model.T), cols::Integer=4)
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

function showlibs{T<:Integer}(model::AbstractCTPF, users::Vector{T})
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

showlibs(model::AbstractCTPF, user::Integer) = showlibs(model, [user])

function showdrecs{T<:Integer}(model::AbstractCTPF, docs::Union{T, Vector{T}}, U::Integer=min(16, model.U); cols::Integer=4)
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

function showurecs{T<:Integer}(model::AbstractCTPF, users::Union{T, Vector{T}}, M::Integer=min(10, model.M); cols::Integer=1)
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

