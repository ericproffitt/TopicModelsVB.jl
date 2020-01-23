type HMTM <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	eta::Vector{Float64}
	alpha::Matrix{Float64}
	beta::Matrix{Float64}
	tau::VectorList{Float64}
	gamma::MatrixList{Float64}
	lambda::VectorList{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function HMTM(corp::Corpus, K::Int)
		@assert ispositive(K)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		eta = ones(K)
		alpha = ones(K, K)
		beta = rand(Dirichlet(V, 1.0), K)'
		tau = Vector[ones(K) for _ in 1:M]
		gamma = Matrix[ones(K, K) for _ in 1:M]
		lambda = Vector[ones(K)/K for _ in 1:M]
		phi = Matrix[ones(K, K)/K for _ in 1:M]

		topics = [collect(1:V) for _ in 1:K]

		model = new(K, M, V, N, C, copy(corp), topics, eta, alpha, beta, tau, gamma, lambda, phi)
		updateELBO!(model)
		return model
	end
end

Base.show(io::IO, model::HMTM) = print(io, "Hidden Markov topic model with $(model.K) topics.")

function Elogppi(model::HMTM, d::Int)
	x = lgamma(sum(model.eta)) - sum(lgamma(model.eta)) + dot(model.eta - 1, digamma(model.tau[d]) - digamma(sum(model.tau[d])))
	return x
end

function Elogptheta(model::HMTM, d::Int)
	x = 0
	for l in 1:model.K
		x += lgamma(sum(model.alpha[:,l])) - sum(lgamma(model.alpha[:,l])) + dot(model.alpha[:,l] - 1, digamma(model.gamma[d][:,l]) - digamma(sum(model.gamma[d][:,l])))
	end
	return x
end

function Elogpz(model::HMTM, d::Int)
	x = dot(model.lambda[d], digamma(model.tau[d]) - digamma(sum(model.tau[d])))
	x += dot(sum([model.phi[d]^(n-2) for n in 2:model.N[d]]) * model.lambda[d], vec(sum(model.phi[d] .* digamma(model.gamma[d]), 1) - digamma(sum(model.gamma[d], 1))))
	return x
end

function Elogpw(model::HMTM, d::Int)
	terms = model.corp[d].terms
	x = 0
	for (n, j) in enumerate(terms)
		x += dot(model.phi[d]^(n-1) * model.lambda[d], log(model.beta[:,j] + eps(0.0)))
	end
	return x
end

function Elogqpi(model::HMTM, d::Int)
	x = -entropy(Dirichlet(model.tau[d]))
	return x
end

function Elogqtheta(model::HMTM, d::Int)
	x = -sum([entropy(Dirichlet(model.gamma[d][:,l])) for l in 1:model.K])
	return x
end

function Elogqz(model::HMTM, d::Int)
	x = sum(log(model.lambda[d].^model.lambda[d]))#-entropy(Categorical(model.lambda[d]))
	x += dot(sum([model.phi[d]^(n-2) for n in 2:model.N[d]]) * model.lambda[d], [sum(log(model.phi[d][:,l].^model.phi[d][:,l])) for l in 1:model.K])
	#x += dot(sum([model.phi[d]^(n-2) for n in 2:model.N[d]]) * model.lambda[d], [-entropy(Categorical(model.phi[d][:,l])) for l in 1:model.K])
	return x
end

function updateELBO!(model::HMTM)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogppi(model, d)
					+ Elogptheta(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d)
					- Elogqpi(model, d)
					- Elogqtheta(model, d)
					- Elogqz(model, d))
	end
	return model.elbo
end

function updateEta!(model::HMTM, niter::Int, ntol::Real)
	nu = model.K
	for _ in 1:niter
		rho = 1.0
		etagrad = [(nu / model.eta[i]) + model.M * (digamma(sum(model.eta)) - digamma(model.eta[i])) + sum([digamma(model.tau[d][i]) - digamma(sum(model.tau[d])) for d in 1:model.M]) for i in 1:model.K]
		etahessdiag = -(model.M * trigamma(model.eta) + (nu ./ model.eta.^2))
		z = model.M * trigamma(sum(model.eta))
		c = sum(etagrad ./ etahessdiag) / (1/z + sum(1 ./ etahessdiag))
		p = ((etagrad - c) ./ etahessdiag)
		
		while minimum(model.eta - rho * p) < 0
			rho *= 0.5
		end	
		model.eta -= rho * p
		
		if (norm(etagrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
	end
end

function updateAlpha!(model::HMTM, niter::Int, ntol::Real)
	for l in 1:model.K
		nu = model.K
		for _ in 1:niter
			rho = 1.0
			alphagrad = [(nu / model.alpha[i,l]) + model.M * (digamma(sum(model.alpha[:,l])) - digamma(model.alpha[i,l])) + sum([digamma(model.gamma[d][i,l]) - digamma(sum(model.gamma[d][:,l])) for d in 1:model.M]) for i in 1:model.K]
			alphahessdiag = -(model.M * trigamma(model.alpha[:,l]) + (nu ./ model.alpha[:,l].^2))
			z = model.M * trigamma(sum(model.alpha))
			c = sum(alphagrad ./ alphahessdiag) / (1/z + sum(1 ./ alphahessdiag))
			p = ((alphagrad - c) ./ alphahessdiag)
			
			while minimum(model.alpha[:,l] - rho * p) < 0
				rho *= 0.5
			end	
			model.alpha[:,l] -= rho * p
			
			if (norm(alphagrad) < ntol) & ((nu / model.K) < ntol)
				break
			end
			nu *= 0.5
		end
	end
end

function updateBeta!(model::HMTM)	
	model.beta = zeros(model.K, model.V)
	for d in 1:model.M
		terms = model.corp[d].terms
		for (n, j) in enumerate(terms)
			model.beta[:,j] += model.phi[d]^(n-1) * model.lambda[d] # shouldn't this be model.phi[d]^(n-1) ???
		end															# ended up changing model.phi[d]^(j-1) to model.phi[d]^(n-1)
	end																# hopefully that's correct
	model.beta ./= sum(model.beta, 2)
end

function updateTau!(model::HMTM, d::Int)
	model.tau[d] = model.eta + model.lambda[d]
end

function updateGamma!(model::HMTM, d::Int)
	model.gamma[d] = model.alpha + model.phi[d] .* (sum([model.phi[d]^(n-2) for n in 2:model.N[d]]) * model.lambda[d])' 
end

function updateLambda!(model::HMTM, d::Int, niter::Int, ntol::Real)
	terms = model.corp[d].terms

	c = vec(digamma(model.tau[d]') - digamma(sum(model.tau[d]))
		+ (sum(model.phi[d] .* digamma(model.gamma[d]), 1) - digamma(sum(model.gamma[d], 1))) * sum([model.phi[d]^(n-2) for n in 2:model.N[d]])
		+ sum([log(model.beta[:,j] + eps(0.0))' * model.phi[d]^(n-1) for (n, j) in enumerate(terms)])
		- sum(log(model.phi[d].^model.phi[d]), 1) * sum([model.phi[d]^(n-2) for n in 2:model.N[d]]))

	v = abs(minimum(c))
	for _ in 1:niter
		v += 1/sum(exp(c + v - 1)) - 1
		if sum(exp(c + v - 1)) - 1 < ntol
			break
		end
	end
	model.lambda[d] = exp(c + v - 1)
end

function updatePhi!(model::HMTM, d::Int)
end

function train!(model::HMTM; iter::Int=200, tol::Real=1.0, niter=1000, ntol::Real=1/model.K^2, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))

	for p in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				oldgamma = model.gamma[d]
				updateTau!(model, d)
				updateGamma!(model, d)
				updatePhi!(model, d)
				updateLambda!(model, d, niter, ntol)
				if norm(oldgamma - model.gamma[d]) < vtol
					break
				end
			end
		end
		updateEta!(model, niter, ntol)
		updateAlpha!(model, niter, ntol)
		updateBeta!(model)
		if checkELBO!(model, p, chkelbo, tol)
			break
		end
	end
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

