type LDA <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float64}
	beta::Matrix{Float64}
	gamma::VectorList{Float64}
	phi::MatrixList{Float64}
	Elogtheta::VectorList{Float64}
	elbo::Float64

	function LDA(corp::Corpus, K::Integer)
		@assert ispositive(K)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		gamma = [ones(K) for _ in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:M]
		Elogtheta = [digamma(ones(K)) - digamma(K) for d in 1:M]

		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, gamma, phi, Elogtheta)
		updateELBO!(model)
		return model
	end
end

function Elogptheta(model::LDA, d::Int)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, model.Elogtheta[d])
	return x
end

function Elogpz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::LDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log(model.beta[:,terms] + epsln) * counts)
	return x
end

function Elogqtheta(model::LDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::LDA)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogptheta(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d) 
					- Elogqtheta(model, d)
					- Elogqz(model, d))
	end
	return model.elbo
end

function updateAlpha!(model::LDA, niter::Integer, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphaGrad = [(nu / model.alpha[i]) + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] + sum(model.Elogtheta)
		alphaHessDiag = -(model.M * trigamma(model.alpha) + (nu ./ model.alpha.^2))
		p = (alphaGrad - sum(alphaGrad ./ alphaHessDiag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(1 ./ alphaHessDiag))) ./ alphaHessDiag
		
		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alphaGrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
	end
	@bumper model.alpha
end

function updateBeta!(model::LDA)	
	model.beta = zeros(model.K, model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.beta[:,terms] += model.phi[d] .* counts'		
	end
	model.beta ./= sum(model.beta, 2)
end

function updateGamma!(model::LDA, d::Int)
	counts = model.corp[d].counts
	@bumper model.gamma[d] = model.alpha + model.phi[d] * counts
end

function updatePhi!(model::LDA, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = model.beta[:,terms] .* exp(model.Elogtheta[d])
	model.phi[d] ./= sum(model.phi[d], 1)
end

function updateElogtheta!(model::LDA, d::Int)
	model.Elogtheta[d] = digamma(model.gamma[d]) - digamma(sum(model.gamma[d]))
end

function train!(model::LDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)
	
	for k in 1:iter
		for d in 1:model.M		
			for _ in 1:viter
				oldgamma = model.gamma[d]
				updateGamma!(model, d)
				updateElogtheta!(model, d)
				updatePhi!(model, d)
				if norm(oldgamma - model.gamma[d]) < vtol
					break
				end
			end
		end	
		updateAlpha!(model, niter, ntol)
		updateBeta!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

