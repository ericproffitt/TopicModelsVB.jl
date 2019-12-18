mutable struct CTM <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float64}
	sigma::Matrix{Float64}
	invsigma::Matrix{Float64}
	beta::Matrix{Float64}
	beta_old::Matrix{Float64}
	beta_temp::Matrix{Float64}
	lambda::VectorList{Float64}
	lambda_old::VectorList{Float64}
	vsq::VectorList{Float64}
	lzeta::Float64
	phi::Matrix{Float64}
	elbo::Float64

	function CTM(corp::Corpus, K::Integer)
		ispositive(K) || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = copy(beta)
		beta_temp = zeros(K, V)
		lambda = [zeros(K) for _ in 1:M]
		lambda_old = copy(lambda)
		vsq = [ones(K) for _ in 1:M]
		lzeta = 0.5
		phi = ones(K, N[1]) / K
		elbo=0

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, beta, beta_old, beta_temp, lambda, lambda_old, vsq, lzeta, phi, elbo)
		
		for d in 1:model.M
			model.phi = ones(K, N[d]) / K
			model.elbo += Elogpeta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqeta(model, d) - Elogqz(model, d)
		end

		return model
	end
end

function Elogpeta(model::CTM, d::Int)
	"Compute the numerical value for E[log(P(eta))]."

	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::CTM, d::Int)
	"Compute the numerical value for E[log(P(z))]."

	counts = model.corp[d].counts
	x = dot(model.phi' * model.lambda[d], counts) + model.C[d] * model.lzeta
	return x
end

function Elogpw(model::CTM, d::Int)
	"Compute the numerical value for E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::CTM, d::Int)
	"Compute the numerical value for E[log(q(eta))]."

	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::CTM, d::Int)
	"Compute the numerical value for E[log(q(z))]."

	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::CTM)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		terms = model.corp[d].terms
		model.phi = additive_logistic(log.(model.beta_old[:,terms]) .+ model.lambda_old[d], dims=1)
		model.phi ./= sum(model.phi, dims=1)
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

	model.sigma = diagm(sum(model.vsq)) / model.M + Base.cov(hcat(model.lambda...), model.mu, 2, false)
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
	model.beta_temp[:,terms] += model.phi .* counts'
end

function update_lambda!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Update lambda."
	"Newton's method."

	counts = model.corp[d].counts
	for _ in 1:niter
		lambda_grad = model.invsigma * (model.mu - model.lambda[d]) + model.phi * counts - model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.lzeta)
		lambda_invhess = -inv(I + model.C[d] * model.sigma * diagm(exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.lzeta))) * model.sigma
		model.lambda[d] -= lambda_invhess * lambda_grad
		
		if norm(lambda_grad) < ntol
			break
		end
	end
end

function update_vsq!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Update Vsq."
	"Interior-point Newton's method with log-barrier and back-tracking line search."

	for _ in 1:niter
		rho = 1.0
		vsq_grad = -0.5 * (diag(model.invsigma) + model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.lzeta) - 1 ./ model.vsq[d])
		vsq_invhess_diag = -1 ./ (0.25 * model.C[d] * exp.(model.lambda[d] + 0.5 * model.vsq[d] .- model.lzeta) + 0.5 ./ model.vsq[d].^2)
		p = vsq_invhess_diag .* vsq_grad
		
		while minimum(model.vsq[d] - rho * p) <= 0
			rho *= 0.5
		end	
		model.vsq[d] -= rho * p
		
		if norm(vsq_grad) < ntol
			break
		end
	end
	@bumper model.vsq[d]
end

function update_lzeta!(model::CTM, d::Int)
	"Update lzeta."
	"Analytic."

	model.lzeta = logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

function update_phi!(model::CTM, d::Int)
	"Update phi."
	"Analytic."

	terms = model.corp[d].terms
	model.phi = additive_logistic(log.(model.beta[:,terms]) .+ model.lambda[d], dims=1)
end

function train!(model::CTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, check_elbo::Integer=1)
	"Coordinate ascent optimization procedure for correlated topic model variational Bayes algorithm."

	@assert all(.!isnegative.([tol, ntol, vtol]))
	@assert all(ispositive.([iter, niter, viter, chkelbo]))
	
	for k in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				update_phi!(model, d)
				update_lzeta!(model, d)
				update_lambda!(model, d, niter, ntol)
				update_vsq!(model, d, niter, ntol)
				if norm(model.lambda[d] - model.lambda_old[d]) < vtol
					break
				end
			end
			update_beta!(model, d)
		end
		update_beta!(model)
		update_mu!(model)
		update_sigma!(model)

		if k % check_elbo == 0
			delta_elbo = -(model.elbo - update_elbo!(model))
			println(k, " ∆elbo: ", round(delta_elbo, digits=3))

			if abs(delta_elbo) < tol
				break
			end
		end
	end

	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

