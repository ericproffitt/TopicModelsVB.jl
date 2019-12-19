### Model utilites for TopicModelsVB
### Eric Proffitt
### December 3, 2019

showdocs(model::TopicModel, doc_indices::Vector{Int}) = showdocs(model.corp, doc_indices)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, doc_range::UnitRange{Int}) = showdocs(model.corp, collect(doc_range))
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)

getlex(model::TopicModel) = sort(collect(values(model.corp.vocab)))
getusers(model::TopicModel) = sort(collect(values(model.corp.users)))

### Display output for TopicModel objects.
Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::fLDA) = print(io, "Filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::CTM) = print(io, "Correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::fCTM) = print(io, "Filtered correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")
Base.show(io::IO, model::gpuLDA) = print(io, "GPU accelerated latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTM) = print(io, "GPU accelerated correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTPF) = print(io, "GPU accelerated collaborative topic Poisson factorization model with $(model.K) topics.")

function update_buffer!(model::gpuLDA)
	"Update gpuLDA model data in GPU RAM."

	@buffer model.Npsums
	@buffer model.Jpsums
	@buffer model.terms
	@buffer model.counts
	@buffer model.words
	@buffer model.phi
	@buffer model.Elogtheta
end

function update_buffer!(model::gpuCTM)
	"Update gpuCTM model data in GPU RAM."

	@buffer model.C
	@buffer model.Npsums
	@buffer model.Jpsums
	@buffer model.terms
	@buffer model.counts
	@buffer model.words
	@buffer model.newtontemp
	@buffer model.newtongrad
	@buffer model.newtoninvhess
	@buffer model.phi
end

function update_buffer!(model::gpuCTPF)
	"Update gpuCTPF model data in GPU RAM."

	@buffer model.Npsums
	@buffer model.Jpsums
	@buffer model.Rpsums
	@buffer model.Ypsums
	@buffer model.terms
	@buffer model.counts
	@buffer model.words
	@buffer model.readers
	@buffer model.ratings
	@buffer model.views
	@buffer model.phi
	@buffer model.xi
end

function update_host!(model::gpuLDA)
	"Update gpuLDA model data in CPU RAM."

	@host model.alphabuf
	@host model.betabuf
	@host model.gammabuf
	@host model.phibuf
	@host model.Elogthetabuf
	@host model.Elogthetasumbuf
end

function update_host!(model::gpuCTM)
	"Update gpuCTM model data in CPU RAM."

	@host model.mubuf
	@host model.sigmabuf
	@host model.invsigmabuf
	@host model.betabuf
	@host model.lambdabuf
	@host model.vsqbuf
	@host model.lzetabuf
	@host model.phibuf
end

function update_host!(model::gpuCTPF)
	"Update gpuCTPF model data in CPU RAM."

	@host model.alefbuf
	@host model.betbuf
	@host model.gimelbuf
	@host model.daletbuf
	@host model.hebuf
	@host model.vavbuf
	@host model.zayinbuf
	@host model.hetbuf
	@host model.phibuf
	@host model.xibuf
end

function check_delta_elbo(model::TopicModel, check_elbo::Real, k::Int, tol::Real)
	"Check and print value of delta_elbo."
	"If abs(delta_elbo) < tol, terminate algorithm."

	if k % check_elbo == 0
		delta_elbo = -(model.elbo - update_elbo!(model))
		println(k, " ∆elbo: ", round(delta_elbo, digits=3))

		if abs(delta_elbo) < tol
			return true
		end
	end
	false
end

function check_model(model::LDA)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite.(model.alpha))
	@assert all(ispositive.(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gamma[d])) for d in 1:model.M])
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	nothing
end

function check_model(model::fLDA)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert isequal(length(model.alpha), model.K)	
	@assert all(isfinite.(model.alpha))
	@assert all(ispositive.(model.alpha))
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
	@assert all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gamma[d])) for d in 1:model.M])
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])	
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	nothing
end

function check_model(model::CTM)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert all(isfinite.(model.mu))	
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.newbeta), (model.K, model.V))
	@assert isequal(model.newbeta, zeros(model.K, model.V))
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.vsq[d])) for d in 1:model.M])	
	@assert isfinite(model.lzeta)	
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isfinite(model.elbo)
	nothing
end

function check_model(model::fCTM)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert (0 <= model.eta <= 1)	
	@assert all(isfinite.(model.mu))
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
	@assert all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])	
	@assert all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.vsq[d])) for d in 1:model.M])	
	@assert isfinite(model.lzeta)
	@assert isequal(length(model.tau), model.M)
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)	
	@assert isfinite(model.elbo)
	nothing
end

function check_model(model::CTPF)
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
	@assert all(isfinite.(model.alef))
	@assert all(ispositive.(model.alef))
	@assert isequal(length(model.bet), model.K)
	@assert all(isfinite.(model.bet))
	@assert all(ispositive.(model.bet))
	@assert isequal(length(model.gimel), model.M)
	@assert all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gimel[d])) for d in 1:model.M])
	@assert isequal(length(model.dalet), model.K)
	@assert all(isfinite.(model.dalet))
	@assert all(ispositive.(model.dalet))
	@assert isequal(size(model.he), (model.K, model.U))	
	@assert all(isfinite.(model.he))
	@assert all(ispositive.(model.he))
	@assert isequal(length(model.vav), model.K)
	@assert all(isfinite.(model.vav))
	@assert all(ispositive.(model.vav))
	@assert isequal(length(model.zayin), model.M)
	@assert all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.zayin[d])) for d in 1:model.M])
	@assert isequal(length(model.het), model.K)
	@assert all(isfinite.(model.het))
	@assert all(ispositive.(model.het))
	@assert isequal(size(model.phi), (model.K, model.N[1]))
	@assert isprobvec(model.phi, 1)
	@assert isequal(size(model.xi), (2model.K, model.R[1]))
	@assert isprobvec(model.xi, 1)
	@assert isfinite(model.elbo)
	nothing	
end

function check_model(model::gpuLDA)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite.(model.alpha))
	@assert all(ispositive.(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)	
	@assert isequal(length(model.gamma), model.M)
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gamma[d])) for d in 1:model.M])	
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])

	model.Elogtheta = [digamma.(model.gamma[d]) - digamma(sum(model.gamma[d])) for d in model.batches[1]]
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
	for in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	model.device, model.context, model.queue = cl.create_compute_context()

	betaprog = cl.Program(model.context, source=LDA_BETA_cpp) |> cl.build!
	betanormprog = cl.Program(model.context, source=LDA_BETA_NORM_cpp) |> cl.build!
	newbetaprog = cl.Program(model.context, source=LDA_NEWBETA_cpp) |> cl.build!
	gammaprog = cl.Program(model.context, source=LDA_GAMMA_cpp) |> cl.build!
	phiprog = cl.Program(model.context, source=LDA_PHI_cpp) |> cl.build!
	phinormprog = cl.Program(model.context, source=LDA_PHI_NORM_cpp) |> cl.build!
	Elogthetaprog = cl.Program(model.context, source=LDA_ELOGTHETA_cpp) |> cl.build!
	Elogthetasumprog = cl.Program(model.context, source=LDA_ELOGTHETASUM_cpp) |> cl.build!

	model.betakern = cl.Kernel(betaprog, "updateBeta")
	model.betanormkern = cl.Kernel(betanormprog, "normalizeBeta")
	model.newbetakern = cl.Kernel(newbetaprog, "updateNewbeta")
	model.gammakern = cl.Kernel(gammaprog, "updateGamma")
	model.phikern = cl.Kernel(phiprog, "updatePhi")
	model.phinormkern = cl.Kernel(phinormprog, "normalizePhi")
	model.Elogthetakern = cl.Kernel(Elogthetaprog, "updateElogtheta")
	model.Elogthetasumkern = cl.Kernel(Elogthetasumprog, "updateElogthetasum")		

	@buf model.alpha
	@buf model.beta
	@buf model.gamma
	@buf model.Elogthetasum
	@buf model.newbeta
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end

function check_model(model::gpuCTM)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert all(isfinite.(model.mu))	
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.vsq[d])) for d in 1:model.M])	
	@assert all(isfinite.(model.lzeta))	
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])

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
	for in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	model.device, model.context, model.queue = cl.create_compute_context()

	muprog = cl.Program(model.context, source=CTM_MU_cpp) |> cl.build!
	betaprog = cl.Program(model.context, source=CTM_BETA_cpp) |> cl.build!
	betanormprog = cl.Program(model.context, source=CTM_BETA_NORM_cpp) |> cl.build!
	newbetaprog = cl.Program(model.context, source=CTM_NEWBETA_cpp) |> cl.build!
	lambdaprog = cl.Program(model.context, source=CTM_LAMBDA_cpp) |> cl.build!
	vsqprog = cl.Program(model.context, source=CTM_VSQ_cpp) |> cl.build!
	lzetaprog = cl.Program(model.context, source=CTM_LZETA_cpp) |> cl.build!
	phiprog = cl.Program(model.context, source=CTM_PHI_cpp) |> cl.build!
	phinormprog = cl.Program(model.context, source=CTM_PHI_NORM_cpp) |> cl.build!

	model.mukern = cl.Kernel(muprog, "updateMu")
	model.betakern = cl.Kernel(betaprog, "updateBeta")
	model.betanormkern = cl.Kernel(betanormprog, "normalizeBeta")
	model.newbetakern = cl.Kernel(newbetaprog, "updateNewbeta")
	model.lambdakern = cl.Kernel(lambdaprog, "updateLambda")
	model.vsqkern = cl.Kernel(vsqprog, "updateVsq")
	model.lzetakern = cl.Kernel(lzetaprog, "updateLzeta")
	model.phikern = cl.Kernel(phiprog, "updatePhi")
	model.phinormkern = cl.Kernel(phinormprog, "normalizePhi")
		
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

function check_model(model::gpuCTPF)
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
	@assert all(isfinite.(model.alef))
	@assert all(ispositive.(model.alef))
	@assert isequal(length(model.bet), model.K)
	@assert all(isfinite.(model.bet))
	@assert all(ispositive.(model.bet))
	@assert isequal(length(model.gimel), model.M)
	@assert all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gimel[d])) for d in 1:model.M])
	@assert isequal(length(model.dalet), model.K)
	@assert all(isfinite.(model.dalet))
	@assert all(ispositive.(model.dalet))
	@assert isequal(size(model.he), (model.K, model.U))	
	@assert all(isfinite.(model.he))
	@assert all(ispositive.(model.he))
	@assert isequal(length(model.vav), model.K)
	@assert all(isfinite.(model.vav))
	@assert all(ispositive.(model.vav))
	@assert isequal(length(model.zayin), model.M)
	@assert all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.zayin[d])) for d in 1:model.M])
	@assert isequal(length(model.het), model.K)
	@assert all(isfinite.(model.het))
	@assert all(ispositive.(model.het))
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])
	@assert isequal(length(model.xi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in model.batches[1]])

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
	for in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	Y = [zeros(Int, model.U) for _ in 1:model.B]
	for in 1:model.B
		for r in model.readers[b]
			Y[b][r+1] += 1
		end
	end

	model.Ypsums = [zeros(Int, model.U + 1) for _ in 1:model.B]
	for in 1:model.B
		for u in 1:model.U
			model.Ypsums[b][u+1] = model.Ypsums[b][u] + Y[b][u]
		end
	end

	model.device, model.context, model.queue = cl.create_compute_context()		

	alefprog = cl.Program(model.context, source=CTPF_ALEF_cpp) |> cl.build!
	newalefprog = cl.Program(model.context, source=CTPF_NEWALEF_cpp) |> cl.build!
	betprog = cl.Program(model.context, source=CTPF_BET_cpp) |> cl.build!
	gimelprog = cl.Program(model.context, source=CTPF_GIMEL_cpp) |> cl.build!
	daletprog = cl.Program(model.context, source=CTPF_DALET_cpp) |> cl.build!
	heprog = cl.Program(model.context, source=CTPF_HE_cpp) |> cl.build!
	newheprog = cl.Program(model.context, source=CTPF_NEWHE_cpp) |> cl.build!
	vavprog = cl.Program(model.context, source=CTPF_VAV_cpp) |> cl.build!
	zayinprog = cl.Program(model.context, source=CTPF_ZAYIN_cpp) |> cl.build!
	hetprog = cl.Program(model.context, source=CTPF_HET_cpp) |> cl.build!
	phiprog = cl.Program(model.context, source=CTPF_PHI_cpp) |> cl.build!
	phinormprog = cl.Program(model.context, source=CTPF_PHI_NORM_cpp) |> cl.build!
	xiprog = cl.Program(model.context, source=CTPF_XI_cpp) |> cl.build!
	xinormprog = cl.Program(model.context, source=CTPF_XI_NORM_cpp) |> cl.build!

	model.alefkern = cl.Kernel(alefprog, "updateAlef")
	model.newalefkern = cl.Kernel(newalefprog, "updateNewalef")
	model.betkern = cl.Kernel(betprog, "updateBet")
	model.gimelkern = cl.Kernel(gimelprog, "updateGimel")
	model.daletkern = cl.Kernel(daletprog, "updateDalet")
	model.hekern = cl.Kernel(heprog, "updateHe")
	model.newhekern = cl.Kernel(newheprog, "updateNewhe")
	model.vavkern = cl.Kernel(vavprog, "updateVav")
	model.zayinkern = cl.Kernel(zayinprog, "updateZayin")
	model.hetkern = cl.Kernel(hetprog, "updateHet")
	model.phikern = cl.Kernel(phiprog, "updatePhi")
	model.phinormkern = cl.Kernel(phinormprog, "normalizePhi")
	model.xikern = cl.Kernel(xiprog, "updateXi")
	model.xinormkern = cl.Kernel(xinormprog, "normalizeXi")
		
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

function gendoc(model::AbstractLDA, laplace_smooth::Real=0.0)
	"Generate artificial document from LDA or gpuLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
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

function gendoc(model::AbstractfLDA, laplace_smooth::Real=0.0)
	"Generate artificial document from fLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
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

function gendoc(model::AbstractCTM, laplace_smooth::Real=0.0)
	"Generate artificial document from CTM or gpuCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp.(theta) / sum(exp.(theta))
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

function gendoc(model::AbstractfCTM, laplace_smooth::Real=0.0)
	"Generate artificial document from fCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
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

function gencorp(model::Union{AbstractLDA, AbstractfLDA, AbstractCTM, AbstractfCTM}, corp_size::Integer, laplace_smooth::Real=0.0)
	"Generate artificial corpus using specified generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	corp_size > 0 || throw(ArgumentError("corp_size parameter must be a positive integer."))
	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	corp = Corpus(vocab=model.corp.vocab, users=model.corp.users)
	corp.docs = [gendoc(model, laplace_smooth) for d in 1:corp_size]
	return corp
end

function showtopics{T<:Integer}(model::TopicModel, N::Integer=min(15, model.V); topics::Union{T, Vector{T}}=collect(1:model.K), cols::Integer=4)
	@assert checkbounds(Bool, 1:model.V, N)
	@assert checkbounds(Bool, 1:model.K, topics)
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

function showlibs{T<:Integer}(model::AbstractCTPF, users::Vector{T})
	@assert checkbounds(Bool, 1:model.U, users)
	
	for u in users
		@juliadots "user $u\n"
		try if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		catch @juliadots model.corp.users[u] * "\n"
		end
		
		for d in model.libs[u]
			yellow(" • ")
			isempty(model.corp[d].title) ? bold("doc $d\n") : bold("$(model.corp[d].title)\n")
		end
		println()
	end
end

showlibs(model::AbstractCTPF, user::Integer) = showlibs(model, [user])

function showdrecs{T<:Integer}(model::AbstractCTPF, docs::Union{T, Vector{T}}, U::Integer=min(16, model.U); cols::Integer=4)
	@assert checkbounds(Bool, 1:model.M, docs)	
	@assert checkbounds(Bool, 1:model.U, U)
	@assert ispositive(cols)
	isa(docs, Vector) || (docs = [docs])
	corp, drecs, users = model.corp, model.drecs, model.corp.users

	for d in docs
		@juliadots "doc $d\n"
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
	@assert checkbounds(Bool, 1:model.U, users)
	@assert checkbounds(Bool, 1:model.M, M)
	@assert ispositive(cols)
	isa(users, Vector) || (users = [users])
	corp, urecs, docs = model.corp, model.urecs, model.corp.docs

	for u in users
		@juliadots "user $u\n"
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

