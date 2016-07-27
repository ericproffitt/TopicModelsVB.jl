type memfLDA <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float64}
	eta::Float64
	etamem::Float64
	beta::Matrix{Float64}
	betamem::Matrix{Float64}
	fbeta::Matrix{Float64}
	kappa::Vector{Float64}
	kappamem::Vector{Float64}
	gamma::Vector{Float64}
	tau::VectorList{Float64}
	phi::Matrix{Float64}
	Elogtheta::VectorList{Float64}
	elbo::Float64
	elbomem::Float64

	function memfLDA(corp::Corpus, K::Integer)
		@assert ispositive(K)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		eta = 0.95
		etamem = 0.0
		beta = rand(Dirichlet(V, 1.0), K)'
		betamem = zeros(K, V)
		fbeta = copy(beta)
		kappa = rand(Dirichlet(V, 1.0))
		kappamem = zeros(V)
		gamma = ones(K)
		tau = [fill(eta, N[d]) for d in 1:M]
		phi = ones(K, N[1]) / K
		Elogtheta = fill(digamma(ones(K)) - digamma(K), M)
	
		model = new(K, M, V, N, C, copy(corp), topics, alpha, eta, etamem, beta, betamem, fbeta, kappa, kappamem, gamma, tau, phi, Elogtheta, 0, 0)
		for d in 1:M
			model.phi = ones(K, N[d]) / K
			updateELBOMEM!(model, d)
		end
		model.phi = ones(K, N[1]) / K
		updateELBO!(model)
		return model
	end
end

function Elogptheta(model::memfLDA, d::Int)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, model.Elogtheta[d])
	return x
end

function Elogpc(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	y = dot(model.tau[d], counts)
	x = log(model.eta^y * (1 - model.eta)^(model.C[d] - y) + epsln)
	return x
end

function Elogpz(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::memfLDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log(model.beta[:,terms] + epsln) * (model.tau[d] .* counts)) + dot(1 - model.tau[d], log(model.kappa[terms] + epsln))
	return x
end

function Elogqtheta(model::memfLDA)
	x = -entropy(Dirichlet(model.gamma))
	return x
end

function Elogqc(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Bernoulli(model.tau[d][n])) for (n, c) in enumerate(counts)])
	return x
end

function Elogqz(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::memfLDA)
	model.elbo = model.elbomem
	model.elbomem = 0
	return model.elbo
end

function updateELBOMEM!(model::memfLDA, d::Int)
	model.elbomem += (Elogptheta(model, d)
					+ Elogpc(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d) 
					- Elogqtheta(model)
					- Elogqc(model, d)
					- Elogqz(model, d))
end

function updateAlpha!(model::memfLDA, niter::Integer, ntol::Real)
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

function updateEta!(model::memfLDA)
	model.eta = model.etamem / sum(model.C)
	model.etamem = 0
end

function updateEtaMEM!(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	model.etamem += dot(model.tau[d], counts)
end

function updateBeta!(model::memfLDA)	
	model.beta = model.betamem ./ sum(model.betamem, 2)
	model.betamem = zeros(model.K, model.V)
end

function updateBetaMEM!(model::memfLDA, d::Int)	
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.betamem[:,terms] += model.phi .* (model.tau[d] .* counts)'
end

function updateKappa!(model::memfLDA)
	model.kappa = model.kappamem / sum(model.kappamem)
	model.kappamem = zeros(model.V)
end

function updateKappaMEM!(model::memfLDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.kappamem[terms] += (1 - model.tau[d]) .* counts
end

function updateGamma!(model::memfLDA, d::Int)
	counts = model.corp[d].counts
	@bumper model.gamma = model.alpha + model.phi * counts	
end

function updateTau!(model::memfLDA, d::Int)
	terms = model.corp[d].terms
	model.tau[d] = model.eta ./ (model.eta + (1 - model.eta) * (model.kappa[terms] .* vec(prod(model.beta[:,terms].^-model.phi, 1))) + epsln)
end

function updatePhi!(model::memfLDA, d::Int)
	terms = model.corp[d].terms
	model.phi = addlogistic(model.tau[d]' .* log(model.beta[:,terms]) .+ model.Elogtheta[d], 1)
end

function updateElogtheta!(model::memfLDA, d::Int)
	model.Elogtheta[d] = digamma(model.gamma) - digamma(sum(model.gamma))
end

function train!(model::memfLDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)	

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for d in 1:model.M	
			for _ in 1:viter
				oldgamma = copy(model.gamma)
				updatePhi!(model, d)
				updateGamma!(model, d)
				updateElogtheta!(model, d)
				if norm(oldgamma - model.gamma) < vtol
					break
				end
			end
			updateTau!(model, d)
			chk && updateELBOMEM!(model, d)
			updateEtaMEM!(model, d)
			updateBetaMEM!(model, d)
			updateKappaMEM!(model, d)
		end
		updateAlpha!(model, niter, ntol)
		updateEta!(model)	
		updateBeta!(model)
		updateKappa!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	model.gamma = ones(model.K)
	model.phi = ones(model.K, model.N[1]) / model.K
	model.fbeta = model.beta .* (model.kappa' .<= 0)
	model.fbeta ./= sum(model.fbeta, 2)
	model.topics = [reverse(sortperm(vec(model.fbeta[i,:]))) for i in 1:model.K]
	nothing
end

