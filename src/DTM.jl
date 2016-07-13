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
	Eexpbeta::MatrixList{Float64}
	a::Vector{Float64}
	rEexpbeta::MatrixList{Float64}
	lzeta::Vector{Float64}
	delta::Float64
	elbo::Float64

	function DTM(corp::Corpus, K::Int, delta::Real, pmodel::Union{LDA, fLDA, CTM, fCTM}=(lda = LDA(corp, K); train!(lda, iter=100, chkelbo=101); lda))
		@assert ispositive(K)
		@assert isequal(pmodel.K, K)
		@assert isfinite(delta)
		@assert ispositive(delta)
		checkcorp(corp)
		checkmodel(pmodel)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		dchrono = sortperm([doc.stamp for doc in corp])
		t0, tM = corp[dchrono[1]].stamp, corp[dchrono[end]].stamp
		T = convert(Int, ceil((tM - t0) / delta))
		S = [Int[] for _ in 1:T]
		t = 1
		for d in dchrono
			corp[d].stamp <= t0 + t * delta ? push!(S[t], d) : (t += 1; push!(S[t], d))
		end

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
		Eexpbeta = [ones(K, V) for _ in 1:T]
		a = ones(T)
		rEexpbeta = [ones(K, V) for _ in 1:T]
		lzeta = ones(M)

		if isa(pmodel, LDA)
			alpha = [pmodel.alpha + eps(0.0) for _ in 1:T]
			betahat = [log(2 * pmodel.beta + eps(1.0)) + randn(K, V) for _ in 1:T]
			gamma = [pmodel.gamma[d] for d in 1:M]
			phi = [pmodel.phi[d] for d in 1:M]

		elseif isa(pmodel, fLDA)
			alpha = [pmodel.alpha + eps(0.0) for _ in 1:T]
			betahat = [log(2 * pmodel.fbeta + eps(1.0)) + randn(K, V) for _ in 1:T]
			gamma = [pmodel.gamma[d] for d in 1:M]
			phi = [pmodel.phi[d] for d in 1:M]

		elseif isa(pmodel, CTM)
			alpha = [exp(pmodel.mu) / sum(exp(pmodel.mu)) for _ in 1:T]
			betahat = [log(2 * pmodel.beta + eps(1.0)) + randn(K, V) for _ in 1:T]
			gamma = [exp(pmodel.lambda[d]) / sum(exp(pmodel.lambda[d])) for d in 1:M]
			phi = [pmodel.phi[d] for d in 1:M]

		elseif isa(pmodel, fCTM)
			alpha = [exp(pmodel.mu) / sum(exp(pmodel.mu)) for _ in 1:T]
			betahat = [log(2 * pmodel.fbeta + eps(1.0)) + randn(K, V) for _ in 1:T]
			gamma = [exp(pmodel.lambda[d]) / sum(exp(pmodel.lambda[d])) for d in 1:M]
			phi = [pmodel.phi[d] for d in 1:M]
		end

		topics = [pmodel.topics for _ in 1:T]

		model = new(K, M, V, N, C, T, S, copy(corp), topics, sigmasq, alpha, gamma, phi, m0, v0, m, v, bsq, betahat, mbeta0, vbeta0, mbeta, vbeta, Eexpbeta, a, rEexpbeta, lzeta, delta)
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
	x = lgamma(sum(model.alpha[t])) - sum(lgamma(model.alpha[t])) + dot(model.alpha[t] - 1, digamma(model.gamma[d]) - digamma(sum(model.gamma[d])))
	return x
end

function Elogpz(model::DTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, digamma(model.gamma[d]) - digamma(sum(model.gamma[d])))
	return x
end

function Elogpw(model::DTM, t::Int, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* model.mbeta[t][:,terms] * counts) - sum(counts .* model.phi[d]' * sum(exp(model.mbeta[t] + 0.5 * model.vbeta[t] - model.lzeta[d]), 2)) - model.lzeta[d] + 1

	#x = sum(model.phi[d] .* model.mbeta[t][:,terms] * counts) - sum(exp(model.a[t] - model.lzeta[d]) * counts .* model.phi[d]' * sum(model.rEexpbeta[t], 2)) - model.lzeta[d] + 1
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

function updateAlpha!(model::DTM, t::Int, niter::Int, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphagrad = [(nu / model.alpha[t][i]) + length(model.S[t]) * (digamma(sum(model.alpha[t])) - digamma(model.alpha[t][i])) + sum([digamma(model.gamma[d][i]) - digamma(sum(model.gamma[d])) for d in model.S[t]]) for i in 1:model.K]
		alphahessdiag = -(length(model.S[t]) * trigamma(model.alpha[t]) + (nu ./ model.alpha[t].^2))
		p = (alphagrad - sum(alphagrad ./ alphahessdiag) / (1 / (length(model.S[t]) * trigamma(sum(model.alpha[t]))) + sum(1./alphahessdiag))) ./ alphahessdiag
		
		while minimum(model.alpha[t] - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha[t] -= rho * p
		
		if (norm(alphagrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
		@buffer model.alpha[t]
	end	
end

function updateGamma!(model::DTM, t::Int, d::Int)
	counts = model.corp[d].counts
	@buffer model.gamma[d] = model.alpha[t] + model.phi[d] * counts
end

function updatePhi!(model::DTM, t::Int, d::Int)
	terms = model.corp[d].terms
	#model.phi[d] = addlogistic(model.mbeta[t][:,terms] .- sum(exp(model.mbeta[t] + 0.5 * model.vbeta[t] - model.lzeta[d]), 2) .+ digamma(model.gamma[d]) - digamma(sum(model.gamma[d])))

	model.phi[d] = addlogistic(model.mbeta[t][:,terms] .- exp(model.a[t] - model.lzeta[d]) * sum(model.rEexpbeta[t], 2) .+ (digamma(model.gamma[d]) - digamma(sum(model.gamma[d]))), 1)
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
	model.a = [maximum(x[t]) for t in 1:model.T]
	model.rEexpbeta = [exp(x[t] - model.a[t]) for t in 1:model.T]
end

function updateVbeta!(model::DTM)
	@buffer model.v[1] = (model.bsq[1] ./ (model.v0 + model.sigmasq + model.bsq[1])) .* (model.v0 + model.sigmasq)
	for t in 2:model.T
		@buffer model.v[t] = (model.bsq[t] ./ (model.v[t-1] + model.sigmasq + model.bsq[t])) .* (model.v[t-1] + model.sigmasq)
	end

	@buffer model.vbeta[model.T] = model.v[model.T]
	for t in model.T:-1:2
		@buffer model.vbeta[t-1] = model.v[t-1] + (model.v[t-1] ./ (model.v[t-1] + model.sigmasq)).^2 .* (model.vbeta[t] - model.v[t-1] - model.sigmasq)
	end
	@buffer model.vbeta0 = model.v0 + (model.v0 ./ (model.v0 + model.sigmasq)).^2 .* (model.vbeta[1] - model.v0 - model.sigmasq)
end

function updateBetahat!(model::DTM, cgiter::Int, cgtol::Real)
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
	model.lzeta[d] = model.a[t] + log(sum(counts .* model.phi[d]' * sum(model.rEexpbeta[t], 2)) + epsln)
end

function train!(model::DTM; iter::Int=150, tol::Real=1.0, niter=1000, ntol::Real=1/model.K^2, viter::Int=10, vtol::Real=1/model.K^2, cgiter::Int=20, cgtol::Real=1/model.T^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, ntol, vtol, cgtol]))
	@assert all(ispositive([iter, niter, viter, cgiter, chkelbo]))	
	checkmodel(model)

	for k in 1:iter
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
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	model.topics = [[reverse(sortperm(vec(model.mbeta[t][i,:]))) for i in 1:model.K] for t in 1:model.T]
	nothing
end

