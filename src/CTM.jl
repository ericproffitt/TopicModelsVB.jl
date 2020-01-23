mutable struct CTM <: TopicModel
	"Correlated topic model."

	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float64}
	sigma::Symmetric{Float64}
	invsigma::Symmetric{Float64}
	beta::Matrix{Float64}
	beta_old::Matrix{Float64}
	beta_temp::Matrix{Float64}
	lambda::VectorList{Float64}
	lambda_old::VectorList{Float64}
	vsq::VectorList{Float64}
	logzeta::Vector{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function CTM(corp::Corpus, K::Integer)
		check_corp(corp)
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = Symmetric(Matrix{Float64}(I, K, K))
		invsigma = copy(sigma)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = copy(beta)
		beta_temp = zeros(K, V)
		lambda = [zeros(K) for _ in 1:M]
		lambda_old = copy(lambda)
		vsq = [ones(K) for _ in 1:M]
		logzeta = fill(0.5, M)
		phi = [ones(K, N[d]) / K for d in 1:min(M, 1)]
		elbo=0

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, beta, beta_old, beta_temp, lambda, lambda_old, vsq, logzeta, phi, elbo)
		
		for d in 1:model.M
			model.phi[1] = ones(K, N[d]) / K
			model.elbo += Elogpeta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqeta(model, d) - Elogqz(model, d)
		end

		return model
	end
end

function Elogpeta(model::CTM, d::Int)
	"Compute E[log(P(eta))]."

	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::CTM, d::Int)
	"Compute E[log(P(z))]."

	counts = model.corp[d].counts
	x = dot(model.phi[1]' * model.lambda[d], counts) + model.C[d] * model.logzeta[d]
	return x
end

function Elogpw(model::CTM, d::Int)
	"Compute E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[1] .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::CTM, d::Int)
	"Compute E[log(q(eta))]."

	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::CTM, d::Int)
	"Compute E[log(q(z))]."

	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[1][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::CTM)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		terms = model.corp[d].terms
		model.phi[1] = additive_logistic(log.(model.beta_old[:,terms]) .+ model.lambda_old[d], dims=1)
		model.phi[1] ./= sum(model.phi[1], dims=1)
		model.elbo += Elogpeta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqeta(model, d) - Elogqz(model, d)
	end

	return model.elbo
end

function update_mu!(model::CTM)
	"Update mu."
	"Analytic."

	model.mu = sum(model.lambda) / model.M
end

function update_sigma!(model::CTM)
	"Update sigma."
	"Analytic"

	model.sigma = Symmetric((diagm(sum(model.vsq)) + (hcat(model.lambda...) .- model.mu) * (hcat(model.lambda...) .- model.mu)') / model.M)
	model.invsigma = inv(model.sigma)
end

function update_beta!(model::CTM)
	"Reset beta variables."

	model.beta_old = model.beta
	model.beta = model.beta_temp ./ sum(model.beta_temp, dims=2)
	model.beta_temp = zeros(model.K, model.V)
end

function update_beta!(model::CTM, d::Int)
	"Update beta."
	"Analytic."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.beta_temp[:,terms] += model.phi[1] .* counts'
end

function update_lambda!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Update lambda."
	"Newton's method."

	model.lambda_old[d] = model.lambda[d]

	counts = model.corp[d].counts
	for _ in 1:niter
		lambda_grad = model.invsigma * (model.mu - model.lambda[d]) + model.phi[1] * counts - model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.logzeta[d])
		lambda_hess = -1 * (model.invsigma + model.C[d] * diagm(exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.logzeta[d])))
		model.lambda[d] -= lambda_hess \ lambda_grad
		
		if norm(lambda_grad) < ntol
			break
		end
	end
end

function update_vsq!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Update vsq."
	"Interior-point Newton's method with log-barrier and back-tracking line search."

	for _ in 1:niter
		rho = 1.0
		vsq_grad = -0.5 * (diag(model.invsigma) + model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.logzeta[d]) - 1 ./ model.vsq[d])
		vsq_invhess_diag = -1 ./ (0.25 * model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.logzeta[d]) + 0.5 ./ model.vsq[d].^2)
		p = vsq_invhess_diag .* vsq_grad
		
		while minimum(model.vsq[d] - rho * p) <= 0
			rho *= 0.5
		end	
		model.vsq[d] -= rho * p
		
		if norm(vsq_grad) < ntol
			break
		end
	end
	@positive model.vsq[d]
end

function update_logzeta!(model::CTM, d::Int)
	"Update logzeta."
	"Analytic."

	model.logzeta[d] = Distributions.logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

function update_phi!(model::CTM, d::Int)
	"Update phi."
	"Analytic."

	terms = model.corp[d].terms
	model.phi[1] = additive_logistic(log.(model.beta[:,terms]) .+ model.lambda[d], dims=1)
end

function train!(model::CTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for correlated topic model variational Bayes algorithm."

	check_model(model)
	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .>= 0)										|| throw(ArgumentError("Iteration parameters must be nonnegative."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf)	|| throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))
	all([isempty(doc) for doc in model.corp]) && (iter = 0)
	update_elbo!(model)

	for k in 1:iter
		for d in 1:model.M
			for v in 1:viter
				update_phi!(model, d)
				update_logzeta!(model, d)
				update_vsq!(model, d, niter, ntol)
				update_lambda!(model, d, niter, ntol)
				if norm(model.lambda[d] - model.lambda_old[d]) < vtol
					break
				end
			end
			update_beta!(model, d)
		end
		update_beta!(model)
		update_sigma!(model)
		update_mu!(model)
		
		if check_elbo!(model, check_elbo, k, tol)
			break
		end
	end

	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end