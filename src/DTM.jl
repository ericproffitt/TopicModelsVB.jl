type DTM <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	T::Int
	S::VectorList{Int}
	corp::Corpus
	topics::Vector{VectorList{Int}}
	delta::Float64
	sigmasq::Float64
	alpha::VectorList{Float64}
	gamma::VectorList{Float64}
	phi::MatrixList{Float64}
	m0::Matrix{Float64}
	v0::Matrix{Float64}
	m::MatrixList{Float64}
	v::MatrixList{Float64}
	bsq::Vector{Float64}
	betahat::MatrixList{Float64}
	mbeta0::Matrix{Float64}
	vbeta0::Matrix{Float64}
	mbeta::MatrixList{Float64}
	vbeta::MatrixList{Float64}
	lzeta::Vector{Float64}
	Elogtheta::VectorList{Float64}
	Eexpbeta::MatrixList{Float64}
	maxlEexpbeta::Vector{Float64}
	ovflEexpbeta::MatrixList{Float64}
	elbo::Float64

	function DTM(corp::Corpus, K::Integer, delta::Real, basemodel::Union{Void, BaseTopicModel}=nothing)
		@assert ispositive(K)
		@assert isfinite(delta)
		@assert ispositive(delta)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		if isa(basemodel, BaseTopicModel)
			fixmodel!(basemodel)
			@assert basemodel.K = K
			@assert basemodel.M = M
			@assert basemodel.V = V 
		end

		stamps = Float64[doc.stamp for doc in corp]
		@assert all(isfinite(stamps))
		t0 = minimum(stamps)
		tM = maximum(stamps)

		T = convert(Int, ceil((tM - t0) / delta))
		S = [Int[] for _ in 1:T]
		
		t = 1
		for d in sortperm(stamps)
			corp[d].stamp > t0 + t * delta && (t += 1)
			push!(S[t], d)
		end

		if isa(basemodel, AbstractLDA)
			topics = [basemodel.topics for _ in 1:T]
			alpha = [basemodel.alpha for _ in 1:T]
			betahat = [log(@boink basemodel.beta) + randn(K, V) for _ in 1:T]
			gamma = [basemodel.gamma[d] for d in 1:M]

		elseif isa(basemodel, AbstractfLDA)
			topics = [basemodel.topics for _ in 1:T]
			alpha = [basemodel.alpha for _ in 1:T]
			betahat = [log(@boink basemodel.beta) + randn(K, V) for _ in 1:T]
			gamma = [basemodel.gamma[d] for d in 1:M]

		elseif isa(basemodel, AbstractCTM)
			topics = [basemodel.topics for _ in 1:T]
			alpha = [addlogistic(basemodel.mu) for _ in 1:T]
			betahat = [log(@boink basemodel.beta) + randn(K, V) for _ in 1:T]
			gamma = [addlogistic(basemodel.lambda[d]) for d in 1:M]

		elseif isa(basemodel, AbstractfCTM)
			topics = [basemodel.topics for _ in 1:T]
			alpha = [addlogistic(basemodel.mu) for _ in 1:T]
			betahat = [log(@boink basemodel.beta) + randn(K, V) for _ in 1:T]
			gamma = [addlogistic(basemodel.lambda[d]) for d in 1:M]

		else
			topics = [[collect(1:V) for _ in 1:K] for _ in 1:T]
			alpha = [ones(K) for _ in 1:T]
			betahat = [randn(K, V) for _ in 1:T]
			gamma = [ones(K) for _ in 1:M]
		end			
		phi = [ones(K, N[d]) / K for d in 1:M]
		
		sigmasq = 1.0
		v0 = ones(K, V)
		v = [ones(K, V) for _ in 1:T]
		m0 = zeros(K, V)
		m = [zeros(K, V) for _ in 1:T]
		bsq = ones(T)
		vbeta0 = ones(K, V)
		vbeta = [ones(K, V) for _ in 1:T]
		mbeta0 = zeros(K, V)
		mbeta = [zeros(K, V) for _ in 1:T]
		lzeta = ones(M)		

		model = new(K, M, V, N, C, T, S, copy(corp), topics, delta, sigmasq, alpha, gamma, phi, m0, v0, m, v, bsq, betahat, mbeta0, vbeta0, mbeta, vbeta, lzeta)
		fixmodel!(model, check=false)

		updateVbeta!(model)
		updateMbeta!(model)		
		updateELBO!(model)
		return model
	end
end

function Elogpbeta(model::DTM, t::Int)
	if t == 1
		x = -0.5 * model.K * model.V * log(2pi * model.sigmasq) - (0.5 / model.sigmasq) * sum((model.mbeta[t] - model.mbeta0).^2 + model.vbeta[t] + model.vbeta0)
	else
		x = -0.5 * model.K * model.V * log(2pi * model.sigmasq) - (0.5 / model.sigmasq) * sum((model.mbeta[t] - model.mbeta[t-1]).^2 + model.vbeta[t] + model.vbeta[t-1])
	end
	return x
end

function Elogptheta(model::DTM, t::Int, d::Int)
	x = lgamma(sum(model.alpha[t])) - sum(lgamma(model.alpha[t])) + dot(model.alpha[t] - 1, model.Elogtheta[d])
	return x
end

function Elogpz(model::DTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::DTM, t::Int, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* model.mbeta[t][:,terms] * counts) - sum(counts .* model.phi[d]' * sum(exp(model.mbeta[t] + 0.5 * model.vbeta[t] - model.lzeta[d]), 2)) - model.lzeta[d] + 1
	return x
end

function Elogqbeta(model::DTM, t::Int)
	x = -sum([entropy(Normal(model.mbeta[t][i,j], sqrt(model.vbeta[t][i,j]))) for i in 1:model.K, j in 1:model.V])
	return x
end

function Elogqtheta(model::DTM, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::DTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::DTM)
	model.elbo = 0
	for t in 1:model.T
		model.elbo += Elogpbeta(model, t) - Elogqbeta(model, t)
		for d in model.S[t]
			model.elbo += (Elogptheta(model, t, d)
						+ Elogpz(model, d)
						+ Elogpw(model, t, d)
						- Elogqtheta(model, d)
						- Elogqz(model, d))
		end
	end
	return model.elbo
end

function updateAlpha!(model::DTM, t::Int, niter::Integer, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphaGrad = [(nu / model.alpha[t][i]) + length(model.S[t]) * (digamma(sum(model.alpha[t])) - digamma(model.alpha[t][i])) for i in 1:model.K] + sum(model.Elogtheta[model.S[t]])
		alphaInvHessDiag = -1 ./ (length(model.S[t]) * trigamma(model.alpha[t]) + nu ./ model.alpha[t].^2)
		p = (alphaGrad - dot(alphaGrad, alphaInvHessDiag) / (1 / (length(model.S[t]) * trigamma(sum(model.alpha[t]))) + sum(alphaInvHessDiag))) .* alphaInvHessDiag
		
		while minimum(model.alpha[t] - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha[t] -= rho * p
		
		if (norm(alphaGrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
		@bumper model.alpha[t]
	end	
end

function updateGamma!(model::DTM, t::Int, d::Int)
	counts = model.corp[d].counts
	@bumper model.gamma[d] = model.alpha[t] + model.phi[d] * counts
end

function updatePhi!(model::DTM, t::Int, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = addlogistic(model.mbeta[t][:,terms] .- exp(model.maxlEexpbeta[t] - model.lzeta[d]) * sum(model.ovflEexpbeta[t], 2) .+ model.Elogtheta[d], 1)
end

function updateMbeta!(model::DTM)
	q = model.bsq[1] ./ (model.v0 + model.sigmasq + model.bsq[1])
	model.m[1] = q .* model.m0 + (1 - q) .* model.betahat[1]
	for t in 2:model.T
		q = model.bsq[t] ./ (model.v[t-1] + model.sigmasq + model.bsq[t])
		model.m[t] = q .* model.m[t-1] + (1 - q) .* model.betahat[t]
	end

	model.mbeta[model.T] = model.m[model.T]
	for t in model.T:-1:2
		q = model.sigmasq ./ (model.v[t-1] + model.sigmasq)
		model.mbeta[t-1] = q .* model.m[t-1] + (1 - q) .* model.mbeta[t]
	end
	q = model.sigmasq ./ (model.v0 + model.sigmasq)
	model.mbeta0 = q .* model.m0 + (1 - q) .* model.mbeta[1]

	x = Matrix{Float64}[model.mbeta[t] + 0.5 * model.vbeta[t] for t in 1:model.T]
	model.Eexpbeta = [exp(x[t]) for t in 1:model.T]	
	model.maxlEexpbeta = [maximum(x[t]) for t in 1:model.T]
	model.ovflEexpbeta = [exp(x[t] - model.maxlEexpbeta[t]) for t in 1:model.T]
end

function updateVbeta!(model::DTM)
	@bumper model.v[1] = (model.bsq[1] ./ (model.v0 + model.sigmasq + model.bsq[1])) .* (model.v0 + model.sigmasq)
	for t in 2:model.T
		@bumper model.v[t] = (model.bsq[t] ./ (model.v[t-1] + model.sigmasq + model.bsq[t])) .* (model.v[t-1] + model.sigmasq)
	end

	@bumper model.vbeta[model.T] = model.v[model.T]
	for t in model.T:-1:2
		@bumper model.vbeta[t-1] = model.v[t-1] + (model.v[t-1] ./ (model.v[t-1] + model.sigmasq)).^2 .* (model.vbeta[t] - model.v[t-1] - model.sigmasq)
	end
	@bumper model.vbeta0 = model.v0 + (model.v0 ./ (model.v0 + model.sigmasq)).^2 .* (model.vbeta[1] - model.v0 - model.sigmasq)
end

function updateBetahat!(model::DTM, cgiter::Integer, cgtol::Real)
	"Nonlinear conjugate gradient (Polak–Ribière) with adaptive back-tracking line search."

	mgrad = [[zeros(model.K, model.V) for _ in 1:model.T] for _ in 1:model.T]
	mbetagrad = [[zeros(model.K, model.V) for _ in 1:model.T] for _ in 1:model.T]
	betahatgrad = [zeros(model.K, model.V) for _ in 1:model.T]
	oldbetahatgrad = [ones(model.K, model.V) for _ in 1:model.T]
	rho = 1.0
	p = 0

	calcstep(model::DTM) = sum([sum([Elogpw(model, t, d) for d in model.S[t]]) + Elogpbeta(model, t) for t in 1:model.T])

	for cg in 1:cgiter
		for s in 1:model.T				
			s == 1 ? mgrad[s][s] = 1 - model.bsq[s] ./ (model.v0 + model.sigmasq + model.bsq[s]) : mgrad[s][s] = 1 - model.bsq[s] ./ (model.v[s-1] + model.sigmasq + model.bsq[s])

			for t in s+1:model.T
				mgrad[t][s] = (model.bsq[t] ./ (model.v[t-1] + model.sigmasq + model.bsq[t])) .* mgrad[t-1][s]
			end

			mbetagrad[model.T][s] = mgrad[model.T][s]
			for t in model.T:-1:s+1
				q = model.sigmasq ./ (model.v[t-1] + model.sigmasq)
				mbetagrad[t-1][s] = q .* mgrad[t-1][s] + (1 - q) .* mbetagrad[t][s]
			end

			betahatgrad[s] = (model.mbeta[1] - model.mbeta0) .* mbetagrad[1][s]
			for t in 2:model.T
				betahatgrad[s] += (model.mbeta[t] - model.mbeta[t-1]) .* (mbetagrad[t][s] - mbetagrad[t-1][s])
			end
			betahatgrad[s] *= -1 / model.sigmasq

			for t in 1:model.T
				x = model.Eexpbeta[t] .* mbetagrad[t][s]
				for d in model.S[t]
					terms, counts = model.corp[d].terms, model.corp[d].counts
					betahatgrad[s][:,terms] += model.phi[d] .* counts' .* mbetagrad[t][s][:,terms]
					betahatgrad[s] -= exp(-model.lzeta[d]) * sum(model.phi[d] .* counts', 2) .* x
				end
			end
		end

		polakribiere = min(1, max(0, sum(vcat(betahatgrad...) .* (vcat(betahatgrad...) - vcat(oldbetahatgrad...))) / sum(vcat(oldbetahatgrad...).^2)))
		p = betahatgrad + polakribiere * p
		oldbetahat = deepcopy(model.betahat)
		oldstep = calcstep(model)
		step = oldstep
		for _ in 1:10
			model.betahat = oldbetahat + rho * p
			updateMbeta!(model)
			step = calcstep(model)
			
			if step > (oldstep + 1e-4 * rho * sum(vcat(betahatgrad...) .* vcat(p...)))
				break
			end
			rho *= 0.5
		end
		rho *= 2
		oldbetahatgrad = deepcopy(betahatgrad)
	end
end

function updateLzeta!(model::DTM, t::Int, d::Int)
	counts = model.corp[d].counts
	model.lzeta[d] = model.maxlEexpbeta[t] + log(@boink sum(counts .* model.phi[d]' * sum(model.ovflEexpbeta[t], 2)))
end

function train!(model::DTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, cgiter::Integer=20, cgtol::Real=1/model.T^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol, cgtol]))
	@assert all(ispositive([iter, niter, viter, cgiter, chkelbo]))	

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for t in 1:model.T
			for d in model.S[t]
				for _ in 1:viter
					oldgamma = model.gamma[d]
					updateGamma!(model, t, d)
					updatePhi!(model, t, d)
					updateLzeta!(model, t, d)
					if norm(oldgamma - model.gamma[d]) < vtol
						break
					end
				end
			end
			updateAlpha!(model, t, niter, ntol)
		end
		updateBetahat!(model, cgiter, cgtol)
		if checkELBO!(model, k, chk, tol)
			break
		end
	end
	model.topics = [[reverse(sortperm(vec(model.mbeta[t][i,:]))) for i in 1:model.K] for t in 1:model.T]
	nothing
end

