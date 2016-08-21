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
	phi::Matrix{Float64}
	Elogtheta::Vector{Float64}
	Elogthetasum::Vector{Float64}
	newbeta::Matrix{Float64}
	elbo::Float64
	newelbo::Float64

	function LDA(corp::Corpus, K::Integer)
		@assert ispositive(K)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		gamma = [ones(K) for _ in 1:M]
		phi = ones(K, N[1]) / K
	
		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, gamma, phi)
		fixmodel!(model, check=false)

		model.newelbo = 0
		for d in 1:M
			model.phi = ones(K, N[d]) / K
			updateNewELBO!(model, d)
		end
		model.phi = ones(K, N[1]) / K
		updateELBO!(model)
		return model
	end
end

function Elogptheta(model::LDA)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, model.Elogtheta)
	return x
end

function Elogpz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi * counts, model.Elogtheta)
	return x
end

function Elogpw(model::LDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqtheta(model::LDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::LDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::LDA)
	model.elbo = model.newelbo
	model.newelbo = 0
	return model.elbo
end

function updateNewELBO!(model::LDA, d::Int)
	model.newelbo += (Elogptheta(model)
					+ Elogpz(model, d)
					+ Elogpw(model, d) 
					- Elogqtheta(model, d)
					- Elogqz(model, d))
end

function updateAlpha!(model::LDA, niter::Integer, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphaGrad = [nu / model.alpha[i] + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] + model.Elogthetasum
		alphaInvHessDiag = -1 ./ (model.M * trigamma(model.alpha) + nu ./ model.alpha.^2)
		p = (alphaGrad - dot(alphaGrad, alphaInvHessDiag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(alphaInvHessDiag))) .* alphaInvHessDiag
		
		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alphaGrad) < ntol) & (nu / model.K < ntol)
			break
		end
		nu *= 0.5
	end
	@bumper model.alpha
	model.Elogthetasum = zeros(model.K)	
end

function updateBeta!(model::LDA)
	model.beta = model.newbeta ./ sum(model.newbeta, 2)
	model.newbeta = zeros(model.K, model.V)
end

function updateNewBeta!(model::LDA, d::Int)	
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.newbeta[:,terms] += model.phi .* counts'		
end

function updateGamma!(model::LDA, d::Int)
	counts = model.corp[d].counts
	@bumper model.gamma[d] = model.alpha + model.phi * counts	
end

function updatePhi!(model::LDA, d::Int)
	terms = model.corp[d].terms
	model.phi = model.beta[:,terms] .* exp(model.Elogtheta)
	model.phi ./= sum(model.phi, 1)
end

function updateElogtheta!(model::LDA, d::Int)
	model.Elogtheta = digamma(model.gamma[d]) - digamma(sum(model.gamma[d]))
end

function updateElogthetasum!(model::LDA)
	model.Elogthetasum += model.Elogtheta
end

function train!(model::LDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for d in 1:model.M	
			for _ in 1:viter
				oldgamma = model.gamma[d]
				updateElogtheta!(model, d)
				updatePhi!(model, d)
				updateGamma!(model, d)
				if norm(oldgamma - model.gamma[d]) < vtol
					break
				end
			end
			updateElogtheta!(model, d)
			updateElogthetasum!(model)
			updateNewBeta!(model, d)
			chk && updateNewELBO!(model, d)
		end
		updateAlpha!(model, niter, ntol)
		updateBeta!(model)
		if checkELBO!(model, k, chk, tol)
			break
		end
	end
	updatePhi!(model, 1)
	updateElogtheta!(model, 1)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end















