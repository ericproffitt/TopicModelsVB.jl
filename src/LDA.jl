"""
    LDA <: TopicModel

Latent Dirichlet allocation model.
"""
mutable struct LDA <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float64}
	beta::Matrix{Float64}
	beta_old::Matrix{Float64}
	beta_temp::Matrix{Float64}
	Elogtheta::VectorList{Float64}
	Elogtheta_old::VectorList{Float64}
	gamma::VectorList{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function LDA(corp::Corpus, K::Integer)
		check_corp(corp)
		K > 0 || throw(ArgumentError("number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = copy(beta)
		beta_temp = zeros(K, V)
		Elogtheta = [-eulergamma * ones(K) .- digamma(K) for _ in 1:M]
		Elogtheta_old = deepcopy(Elogtheta)
		gamma = [ones(K) for _ in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:min(M, 1)]
		elbo = 0
	
		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, beta_old, beta_temp, Elogtheta, Elogtheta_old, gamma, phi, elbo)
		return model
	end
end

## Compute E_q[log(P(theta))].
function Elogptheta(model::LDA, d::Int)
	x = finite(loggamma(sum(model.alpha))) - finite(sum(loggamma.(model.alpha))) + dot(model.alpha .- 1, model.Elogtheta[d])
	return x
end

## Compute E_q[log(P(z))].
function Elogpz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[1] * counts, model.Elogtheta[d])
	return x
end

## Compute E_q[log(P(w))].
function Elogpw(model::LDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[1] .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

## Compute E_q[log(q(theta))].
function Elogqtheta(model::LDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

## Compute E_q[log(q(z))].
function Elogqz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[1][:,n])) for (n, c) in enumerate(counts)])
	return x
end

## Update evidence lower bound.
function update_elbo!(model::LDA)
	model.elbo = 0
	for d in 1:model.M
		terms = model.corp[d].terms
		@positive model.phi[1] = model.beta_old[:,terms] .* exp.(model.Elogtheta_old[d])
		model.phi[1] ./= sum(model.phi[1], dims=1)
		model.elbo += Elogptheta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqtheta(model, d) - Elogqz(model, d)
	end

	return model.elbo
end

## Update alpha.
## Interior-point Newton's method with log-barrier and back-tracking line search.
function update_alpha!(model::LDA, niter::Integer, ntol::Real)
	Elogtheta_sum = sum([model.Elogtheta[d] for d in 1:model.M])

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alpha_grad = nu ./ model.alpha + model.M * (digamma(sum(model.alpha)) .- digamma.(model.alpha)) + Elogtheta_sum
		h_inv = -1 ./ (model.M * trigamma.(model.alpha) + nu ./ model.alpha.^2)
		p = (alpha_grad .- dot(alpha_grad, h_inv) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(h_inv))) .* h_inv
		
		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		@finite model.alpha -= rho * p
		
		if (rho * norm(alpha_grad) < ntol) & (nu / model.K < ntol)
			break
		end
		nu *= 0.5
	end
	@positive model.alpha
end

## Reset beta variables.
function update_beta!(model::LDA)
	model.beta_old = model.beta
	model.beta = model.beta_temp ./ sum(model.beta_temp, dims=2)
	model.beta_temp = zeros(model.K, model.V)
end

## Update beta.
## Analytic.
function update_beta!(model::LDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.beta_temp[:,terms] += model.phi[1] .* counts'		
end

## Update E[log(theta)].
## Analytic.
function update_Elogtheta!(model::LDA, d::Int)
	model.Elogtheta_old[d] = model.Elogtheta[d]
	model.Elogtheta[d] = digamma.(model.gamma[d]) .- digamma(sum(model.gamma[d]))
end

## Update gamma.
## Analytic.
function update_gamma!(model::LDA, d::Int)
	counts = model.corp[d].counts
	@positive model.gamma[d] = model.alpha + model.phi[1] * counts
end

## Update phi.
## Analytic.
function update_phi!(model::LDA, d::Int)
	terms = model.corp[d].terms
	@positive model.phi[1] = model.beta[:,terms] .* exp.(model.Elogtheta[d])
	model.phi[1] ./= sum(model.phi[1], dims=1)
end

"""
    train!(model::LDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, checkelbo::Real=1, printelbo::Bool=true)

Coordinate ascent optimization procedure for latent Dirichlet allocation variational Bayes algorithm.
"""
function train!(model::LDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, checkelbo::Real=1, printelbo::Bool=true)
	check_model(model)
	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .>= 0)										|| throw(ArgumentError("iteration parameters must be nonnegative."))
	(isa(checkelbo, Integer) & (checkelbo > 0)) | (checkelbo == Inf)	|| throw(ArgumentError("checkelbo parameter must be a positive integer or Inf."))
	all([isempty(doc) for doc in model.corp]) && (iter = 0)
	(checkelbo <= iter) && update_elbo!(model)

	for k in 1:iter
		for d in 1:model.M	
			for v in 1:viter
				update_phi!(model, d)
				update_gamma!(model, d)
				update_Elogtheta!(model, d)
				if norm(model.Elogtheta[d] - model.Elogtheta_old[d]) < vtol
					break
				end
			end
			update_beta!(model, d)
		end
		update_beta!(model)
		update_alpha!(model, niter, ntol)
		
		if check_elbo!(model, checkelbo, printelbo, k, tol)
			break
		end
	end

	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end
