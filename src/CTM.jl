type CTM <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float64}
	sigma::Matrix{Float64}
	beta::Matrix{Float64}
	lambda::VectorList{Float64}
	vsq::VectorList{Float64}
	lzeta::Float64
	phi::Matrix{Float64}
	invsigma::Matrix{Float64}
	newbeta::Matrix{Float64}
	elbo::Float64
	newelbo::Float64

	function CTM(corp::Corpus, K::Integer)
		@assert ispositive(K)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		lambda = [zeros(K) for _ in 1:M]
		vsq = [ones(K) for _ in 1:M]
		lzeta = 0.5
		phi = ones(K, N[1]) / K

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, beta, lambda, vsq, lzeta, phi)
		fixmodel!(model, check=false)

		for d in 1:M
			model.phi = ones(K, N[d]) / K
			updateNewELBO!(model, d)
		end
		model.phi = ones(K, N[1]) / K	
		updateELBO!(model)
		return model
	end
end

function Elogpeta(model::CTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::CTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi' * model.lambda[d], counts) + model.C[d] * model.lzeta
	return x
end

function Elogpw(model::CTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::CTM, d::Int)
	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::CTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::CTM)
	model.elbo = model.newelbo
	model.newelbo = 0
	return model.elbo
end

function updateNewELBO!(model::CTM, d::Int)
	model.newelbo += (Elogpeta(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d)
					- Elogqeta(model, d)
					- Elogqz(model, d))					 
end

function updateMu!(model::CTM)
	model.mu = sum(model.lambda) / model.M
end

function updateSigma!(model::CTM)
	model.sigma = diagm(sum(model.vsq)) / model.M + cov(hcat(model.lambda...)', mean=model.mu', corrected=false)
	(log(cond(model.sigma)) < 14) || (model.sigma += eye(model.K) * (eigmax(model.sigma) - 14 * eigmin(model.sigma)) / 13)
	model.invsigma = inv(model.sigma)
end

function updateBeta!(model::CTM)
	model.beta = model.newbeta ./ sum(model.newbeta, 2)
	model.newbeta = zeros(model.K, model.V)
end

function updateNewBeta!(model::CTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.newbeta[:,terms] += model.phi .* counts'
end

function updateLambda!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	counts = model.corp[d].counts
	for _ in 1:niter
		lambdaGrad = model.invsigma * (model.mu - model.lambda[d]) + model.phi * counts - model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta)
		lambdaInvHess = -inv(eye(model.K) + model.C[d] * model.sigma * diagm(exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta))) * model.sigma
		model.lambda[d] -= lambdaInvHess * lambdaGrad
		if norm(lambdaGrad) < ntol
			break
		end
	end
end

function updateVsq!(model::CTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	for _ in 1:niter
		rho = 1.0
		vsqGrad = -0.5 * (diag(model.invsigma) + model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta) - 1 ./ model.vsq[d])
		vsqInvHess = -1 ./ (0.25 * model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta) + 0.5 ./ model.vsq[d].^2)
		p = vsqInvHess .* vsqGrad
		
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

function updateLzeta!(model::CTM, d::Int)
	model.lzeta = logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

function updatePhi!(model::CTM, d::Int)
	terms = model.corp[d].terms
	model.phi = addlogistic(log(model.beta[:,terms]) .+ model.lambda[d], 1)
end

function train!(model::CTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	
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
			updateNewBeta!(model, d)
			chk && updateNewELBO!(model, d)
		end
		updateMu!(model)
		updateSigma!(model)
		updateBeta!(model)
		if checkELBO!(model, k, chk, tol)
			break
		end
	end
	updatePhi!(model, 1)
	updateLzeta!(model, 1)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

