type memfCTM <: TopicModel
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
	etamem::Float64
	beta::Matrix{Float64}
	betamem::Matrix{Float64}
	fbeta::Matrix{Float64}
	kappa::Vector{Float64}
	kappamem::Vector{Float64}
	lambda::VectorList{Float64}
	vsq::VectorList{Float64}
	lzeta::Float64
	tau::VectorList{Float64}
	phi::Matrix{Float64}
	elbo::Float64
	elbomem::Float64

	function memfCTM(corp::Corpus, K::Integer)
		@assert ispositive(K)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		invsigma = eye(K)
		eta = 0.95
		etamem = 0.0
		beta = rand(Dirichlet(V, 1.0), K)'
		betamem = zeros(K, V)
		fbeta = beta
		kappa = rand(Dirichlet(V, 1.0))
		kappamem = zeros(V)
		lambda = [zeros(K) for _ in 1:M]
		vsq = [ones(K) for _ in 1:M]
		lzeta = 0.0
		tau = [fill(eta, N[d]) for d in 1:M]
		phi = ones(K, N[1]) / K


		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, eta, etamem, beta, betamem, fbeta, kappa, kappamem, lambda, vsq, lzeta, tau, phi, 0, 0)
		for d in 1:M
			model.phi = ones(K, N[d]) / K
			updateELBOMEM!(model, d)
		end
		model.phi = ones(K, N[1]) / K	
		updateELBO!(model)
		return model
	end
end

function Elogpeta(model::memfCTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpc(model::memfCTM, d::Int)
	counts = model.corp[d].counts
	y = dot(model.tau[d], counts)
	x = log(model.eta^y * (1 - model.eta)^(model.C[d] - y))
	return x
end

function Elogpz(model::memfCTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi' * model.lambda[d], counts) + model.C[d] * model.lzeta
	return x
end

function Elogpw(model::memfCTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log(model.beta[:,terms] + epsln) * (model.tau[d] .* counts)) + dot(1 - model.tau[d], log(model.kappa[terms] + epsln))
	return x
end

function Elogqeta(model::memfCTM, d::Int)
	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqc(model::memfCTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Bernoulli(model.tau[d][n])) for (n, c) in enumerate(counts)])
	return x
end

function Elogqz(model::memfCTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::memfCTM)
	model.elbo = model.elbomem
	model.elbomem = 0
	return model.elbo
end

function updateELBOMEM!(model::memfCTM, d::Int)
	model.elbomem += (Elogpeta(model, d)
					+ Elogpc(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d)
					- Elogqeta(model, d)
					- Elogqc(model, d)
					- Elogqz(model, d))					 
end

function updateMu!(model::memfCTM)
	model.mu = sum(model.lambda) / model.M
end

function updateSigma!(model::memfCTM)
	model.sigma = diagm(sum(model.vsq)) / model.M + cov(hcat(model.lambda...)', mean=model.mu', corrected=false)
	(log(cond(model.sigma)) < 14) || (model.sigma += eye(model.K) * (eigmax(model.sigma) - 14 * eigmin(model.sigma)) / 13)
	model.invsigma = inv(model.sigma)
end

function updateEta!(model::memfCTM)
	model.eta = sum([dot(model.tau[d], model.corp[d].counts) for d in 1:model.M]) / sum(model.C)
end

function updateEtaMEM!(model::memfCTM, d::Int)
	counts = model.corp[d].counts
	model.etamem += dot(model.tau[d], counts)
end

function updateBeta!(model::memfCTM)	
	model.beta = model.betamem ./ sum(model.betamem, 2)
	model.betamem = zeros(model.K, model.V)
end

function updateBetaMEM!(model::memfCTM, d::Int)	
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.betamem[:,terms] += model.phi .* (model.tau[d] .* counts)'
end

function updateKappa!(model::memfCTM)
	model.kappa = model.kappamem / sum(model.kappamem)
	model.kappamem = zeros(model.V)
end

function updateKappaMEM!(model::memfCTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.kappamem[terms] += (1 - model.tau[d]) .* counts
end

function updateLambda!(model::memfCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	counts = model.corp[d].counts
	for _ in 1:niter
		lambdaGrad = (-model.invsigma * (model.lambda[d] - model.mu) + model.phi * counts - model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta))
		lambdaHessInv = -inv(eye(model.K) + model.C[d] * model.sigma * diagm(exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta))) * model.sigma
		model.lambda[d] -= lambdaHessInv * lambdaGrad
		if norm(lambdaGrad) < ntol
			break
		end
	end
end

function updateVsq!(model::memfCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	for _ in 1:niter
		rho = 1.0
		vsqGrad = -0.5 * (diag(model.invsigma) + model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta) - 1 ./ model.vsq[d])
		vsqHessInv = -diagm(1 ./ (0.25 * model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta) + 0.5 ./ model.vsq[d].^2))
		p = vsqHessInv * vsqGrad
		
		while minimum(model.vsq[d] - rho * p) <= 0
			rho *= 0.5
		end	
		model.vsq[d] -= rho * p
		
		if norm(vsqGrad) < ntol
			break
		end
	end
	@bumper model.vsq[d]
end

function updateLzeta!(model::memfCTM, d::Int)
	model.lzeta = logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

function updateTau!(model::memfCTM, d::Int)
	terms = model.corp[d].terms
	model.tau[d] = model.eta ./ (model.eta + (1 - model.eta) * (model.kappa[terms] .* vec(prod(model.beta[:,terms].^-model.phi, 1))) + epsln)
end

function updatePhi!(model::memfCTM, d::Int)
	terms = model.corp[d].terms
	model.phi = addlogistic(model.tau[d]' .* log(model.beta[:,terms]) .+ model.lambda[d], 1)
end

function train!(model::memfCTM; iter::Integer=150, tol::Real=1.0, niter=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for d in 1:model.M
			for _ in 1:viter
				oldlambda = model.lambda[d]
				updatePhi!(model, d)
				updateLzeta!(model, d)
				updateLambda!(model, d, niter, ntol)
				updateVsq!(model, d, niter, ntol)
				if norm(oldlambda - model.lambda[d]) < vtol
					break
				end
			end
			updateTau!(model, d)
			chk && updateELBOMEM!(model, d)
			updateEtaMEM!(model, d)
			updateBetaMEM!(model, d)
			updateKappaMEM!(model, d)
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
	model.lzeta = 0.0
	model.phi = ones(model.K, model.N[1]) / model.K
	model.fbeta = model.beta .* (model.kappa' .<= 0)
	model.fbeta ./= sum(model.fbeta, 2)
	model.topics = [reverse(sortperm(vec(model.fbeta[i,:]))) for i in 1:model.K]
	nothing
end

