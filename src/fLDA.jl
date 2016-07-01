type fLDA <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float64}
	eta::Float64
	beta::Matrix{Float64}
	fbeta::Matrix{Float64}
	kappa::Vector{Float64}
	gamma::VectorList{Float64}
	tau::VectorList{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function fLDA(corp::Corpus, K::Int)
		@assert ispositive(K)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		alpha = ones(K)
		eta = 0.95
		beta = rand(Dirichlet(V, 1.0), K)'
		fbeta = beta
		kappa = rand(Dirichlet(V, 1.0))
		gamma = [ones(K) for _ in 1:M]
		tau = [rand(N[d]) for d in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:M]

		topics = [collect(1:V) for _ in 1:K]

		model = new(K, M, V, N, C, copy(corp), topics, alpha, eta, beta, fbeta, kappa, gamma, tau, phi)
		updateELBO!(model)
		return model
	end
end

function Elogptheta(model::fLDA, d::Int)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, digamma(model.gamma[d]) - digamma(sum(model.gamma[d])))
	return x
end

function Elogpc(model::fLDA, d::Int)
	counts = model.corp[d].counts
	a = dot(model.tau[d], counts)
	x = log(model.eta^a * (1 - model.eta)^(model.C[d] - a) + epsln)
	return x
end

function Elogpz(model::fLDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, digamma(model.gamma[d]) - digamma(sum(model.gamma[d])))
	return x
end

function Elogpw(model::fLDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log(model.beta[:,terms] + epsln) * (model.tau[d] .* counts)) + dot(1 - model.tau[d], log(model.kappa[terms] + epsln))
	return x
end

function Elogqtheta(model::fLDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqc(model::fLDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Bernoulli(model.tau[d][n])) for (n, c) in enumerate(counts)])
	return x
end

function Elogqz(model::fLDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::fLDA)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogptheta(model, d)
					+ Elogpc(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d) 
					- Elogqtheta(model, d)
					- Elogqc(model, d)
					- Elogqz(model, d))
	end
	return model.elbo
end

function updateAlpha!(model::fLDA, niter::Int, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphagrad = [(nu / model.alpha[i]) + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) + sum([digamma(model.gamma[d][i]) - digamma(sum(model.gamma[d])) for d in 1:model.M]) for i in 1:model.K]
		alphahessdiag = -(model.M * trigamma(model.alpha) + (nu ./ model.alpha.^2))
		p = (alphagrad - sum(alphagrad ./ alphahessdiag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(1 ./ alphahessdiag))) ./ alphahessdiag
		
		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alphagrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
	end	
	@buffer model.alpha
end

function updateEta!(model::fLDA)
	model.eta = sum([dot(model.tau[d], model.corp[d].counts) for d in 1:model.M]) / sum(model.C)
end

function updateBeta!(model::fLDA)	
	model.beta = zeros(model.K, model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.beta[:,terms] += model.phi[d] .* (model.tau[d] .* counts)'
	end
	model.beta ./= sum(model.beta, 2)
end

function updateKappa!(model::fLDA)
	model.kappa = zeros(model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.kappa[terms] += (1 - model.tau[d]) .* counts
	end
	model.kappa /= sum(model.kappa)
end

function updateGamma!(model::fLDA, d::Int)
	counts = model.corp[d].counts
	@buffer model.gamma[d] = model.alpha + model.phi[d] * counts
end

function updateTau!(model::fLDA, d::Int)
	terms = model.corp[d].terms
	model.tau[d] = model.eta ./ (model.eta + (1 - model.eta) * (model.kappa[terms] .* vec(prod(model.beta[:,terms].^-model.phi[d], 1))) + epsln)
end

function updatePhi!(model::fLDA, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = addlogistic(model.tau[d]' .* log(model.beta[:,terms]) .+ digamma(model.gamma[d]) - digamma(sum(model.gamma[d])), 1)
end

function train!(model::fLDA; iter::Int=200, tol::Real=1.0, niter=1000, ntol::Real=1/model.K^2, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	checkmodel(model)	
	
	for k in 1:iter
		for d in 1:model.M		
			for _ in 1:viter
				oldgamma = model.gamma[d]
				updateGamma!(model, d)
				updatePhi!(model, d)
				if norm(oldgamma - model.gamma[d]) < vtol
					break
				end
			end
			updateTau!(model, d)
		end	
		updateAlpha!(model, niter, ntol)
		updateEta!(model)
		updateBeta!(model)
		updateKappa!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	model.fbeta = model.beta .* (model.kappa' .<= 0)
	model.fbeta ./= sum(model.fbeta, 2)
	model.topics = [reverse(sortperm(vec(model.fbeta[i,:]))) for i in 1:model.K]
	nothing
end

