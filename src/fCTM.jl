type fCTM <: TopicModel
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
	eta::Float64
	beta::Matrix{Float64}
	fbeta::Matrix{Float64}
	kappa::Vector{Float64}
	lambda::VectorList{Float64}
	vsq::VectorList{Float64}
	lzeta::Vector{Float64}
	tau::VectorList{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function fCTM(corp::Corpus, K::Integer)
		@assert ispositive(K)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		invsigma = eye(K)
		eta = 0.95
		beta = rand(Dirichlet(V, 1.0), K)'
		fbeta = beta
		kappa = rand(Dirichlet(V, 1.0))
		lambda = [zeros(K) for _ in 1:M]
		vsq = [ones(K) for _ in 1:M]
		lzeta = zeros(M)
		tau = [fill(eta, N[d]) for d in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:M]

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, eta, beta, fbeta, kappa, lambda, vsq, lzeta, tau, phi)
		updateELBO!(model)
		return model
	end
end

function Elogpeta(model::fCTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpc(model::fCTM, d::Int)
	counts = model.corp[d].counts
	a = dot(model.tau[d], counts)
	x = log(model.eta^a * (1 - model.eta)^(model.C[d] - a))
	return x
end

function Elogpz(model::fCTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d]' * model.lambda[d], counts) + model.C[d] * model.lzeta[d]
	return x
end

function Elogpw(model::fCTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log(model.beta[:,terms] + epsln) * (model.tau[d] .* counts)) + dot(1 - model.tau[d], log(model.kappa[terms] + epsln))
	return x
end

function Elogqeta(model::fCTM, d::Int)
	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqc(model::fCTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Bernoulli(model.tau[d][n])) for (n, c) in enumerate(counts)])
	return x
end

function Elogqz(model::fCTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::fCTM)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogpeta(model, d)
					+ Elogpc(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d)
					- Elogqeta(model, d)
					- Elogqc(model, d)
					- Elogqz(model, d))					 
	end		
	return model.elbo
end

function updateMu!(model::fCTM)
	model.mu = sum(model.lambda) / model.M
end

function updateSigma!(model::fCTM)
	model.sigma = diagm(sum(model.vsq)) / model.M + cov(hcat(model.lambda...)', mean=model.mu', corrected=false)
	(log(cond(model.sigma)) < 14) || (model.sigma += eye(model.K) * (eigmax(model.sigma) - 14 * eigmin(model.sigma)) / 13)
	model.invsigma = inv(model.sigma)
end

function updateEta!(model::fCTM)
	model.eta = sum([dot(model.tau[d], model.corp[d].counts) for d in 1:model.M]) / sum(model.C)
end

function updateBeta!(model::fCTM)	
	model.beta = zeros(model.K, model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.beta[:,terms] += model.phi[d] .* (model.tau[d] .* counts)'
	end
	model.beta ./= sum(model.beta, 2)
end

function updateKappa!(model::fCTM)
	model.kappa = zeros(model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.kappa[terms] += (1 - model.tau[d]) .* counts
	end
	model.kappa /= sum(model.kappa)
end

function updateLambda!(model::fCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	counts = model.corp[d].counts
	for _ in 1:niter
		lambdaGrad = (-model.invsigma * (model.lambda[d] - model.mu) + model.phi[d] * counts 
						- model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]))
		lambdaHessInv = -inv(eye(model.K) + model.C[d] * model.sigma * diagm(exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]))) * model.sigma
		model.lambda[d] -= lambdaHessInv * lambdaGrad
		if norm(lambdaGrad) < ntol
			break
		end
	end
end

function updateVsq!(model::fCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	for _ in 1:niter
		rho = 1.0
		vsqGrad = -0.5 * (diag(model.invsigma) + model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]) - 1 ./ model.vsq[d])
		vsqHessInv = -diagm(1 ./ (0.25 * model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]) + 0.5 ./ model.vsq[d].^2))
		p = vsqHessInv * vsqGrad
		
		while minimum(model.vsq[d] - rho * p) < 0
			rho *= 0.5
		end	
		model.vsq[d] -= rho * p
		
		if norm(vsqGrad) < ntol
			break
		end
	end
	@bumper model.vsq[d]
end

function updateLzeta!(model::fCTM, d::Int)
	model.lzeta[d] = logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

function updateTau!(model::fCTM, d::Int)
	terms = model.corp[d].terms
	model.tau[d] = model.eta ./ (model.eta + (1 - model.eta) * (model.kappa[terms] .* vec(prod(model.beta[:,terms].^-model.phi[d], 1))) + epsln)
end

function updatePhi!(model::fCTM, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = addlogistic(model.tau[d]' .* log(model.beta[:,terms]) .+ model.lambda[d], 1)
end

function train!(model::fCTM; iter::Integer=150, tol::Real=1.0, niter=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)

	for k in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				oldlambda = model.lambda[d]
				updateLambda!(model, d, niter, ntol)
				updateVsq!(model, d, niter, ntol)
				updateLzeta!(model, d)
				updatePhi!(model, d)
				if norm(oldlambda - model.lambda[d]) < vtol
					break
				end
			end
			updateTau!(model, d)
		end
		updateMu!(model)
		updateSigma!(model)
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

