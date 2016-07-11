type CTPF <: TopicModel
	K::Int
	M::Int
	V::Int
	U::Int
	N::Vector{Int}
	C::Vector{Int}
	R::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	scores::Matrix{Float64}
	libs::VectorList{Int}
	drecs::VectorList{Int}
	urecs::VectorList{Int}
	a::Float64
	b::Float64
	c::Float64
	d::Float64
	e::Float64
	f::Float64
	g::Float64
	h::Float64
	alef::Matrix{Float64}
	bet::Matrix{Float64}
	alefbet::Vector{Float64}
	gimel::VectorList{Float64}
	dalet::VectorList{Float64}
	he::Matrix{Float64}
	vav::Matrix{Float64}
	hevav::Vector{Float64}
	zayin::VectorList{Float64}
	het::VectorList{Float64}
	phi::MatrixList{Float64}
	xi::MatrixList{Float64}
	elbo::Float64

	function CTPF(corp::Corpus, K::Int, pmodel::Union{LDA, fLDA, CTM, fCTM}=(lda = LDA(corp, K); train!(lda, iter=150, chkelbo=151); lda))
		@assert ispositive(K)		
		@assert isequal(pmodel.K, K)
		checkcorp(corp)
		checkmodel(pmodel)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		R = [length(doc.readers) for doc in corp]

		@assert isequal(collect(1:U), sort(collect(keys(corp.users))))
		libs = [Int[] for _ in 1:U]
		for u in 1:U, d in 1:M
			if u in corp[d].readers; push!(libs[u], d); end
		end

		a, b, c, d, e, f, g, h = fill(0.1, 8)

		alef = exp(pmodel.beta - 0.5)
		bet = ones(K, V)
		alefbet = vec(sum(alef ./ bet, 2))
		gimel = deepcopy(pmodel.gamma)
		dalet = [ones(K) for _ in 1:M]
		he = ones(K, U)
		vav = ones(K, U)
		hevav = vec(sum(he ./ vav, 2))
		zayin = [ones(K) for _ in 1:M]
		het = [ones(K) for _ in 1:M]

		phi = [rand(Dirichlet(K, 1.0), N[d]) for d in 1:M]
		xi = [rand(Dirichlet(2K, 1.0), R[d]) for d in 1:M]

		topics = [collect(1:V) for _ in 1:K]

		model = new(K, M, V, U, N, C, R, copy(corp), topics, zeros(M, U), libs, Vector[], Vector[], a, b, c, d, e, f, g, h, alef, bet, alefbet, gimel, dalet, he, vav, hevav, zayin, het, phi, xi)
		updateELBO!(model)
		return model
	end
end

function Elogpya(model::CTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[d][i,u])
		x += (ra * model.xi[d][i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[d][i]) + digamma(model.he[i,re]) - log(model.vav[i,re]))
				- (model.gimel[d][i] / model.dalet[d][i]) * (model.he[i,re] / model.vav[i,re]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::CTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[d][model.K+i,u])
		x += (ra * model.xi[d][model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[d][i]) + digamma(model.he[i,re]) - log(model.vav[i,re]))
				- (model.zayin[d][i] / model.het[d][i]) * (model.he[i,re] / model.vav[i,re]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::CTPF, d::Int)
	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[d][i,n])
		x += (c * model.phi[d][i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[d][i]) + digamma(model.alef[i,j]) - log(model.bet[i,j]))
				- (model.gimel[d][i] / model.dalet[d][i]) * (model.alef[i,j] / model.bet[i,j]) - sum([pdf(binom, z) * lgamma(z + 1) for z in 0:c]))
	end
	return x
end

function Elogpbeta(model::CTPF)
	x = model.V * model.K * (model.a * log(model.b) - lgamma(model.a))
	for j in 1:model.V, i in 1:model.K
		x += (model.a - 1) * (digamma(model.alef[i,j]) - log(model.bet[i,j])) - model.b * model.alef[i,j] / model.bet[i,j]
	end
	return x
end

function Elogptheta(model::CTPF, d::Int)
	x = model.K * (model.c * log(model.d) - lgamma(model.c))
	for i in 1:model.K
		x += (model.c - 1) * (digamma(model.gimel[d][i]) - log(model.dalet[d][i])) - model.d * model.gimel[d][i] / model.dalet[d][i]
	end
	return x
end

function Elogpeta(model::CTPF)
	x = model.U * model.K * (model.e * log(model.f) - lgamma(model.e))
	for u in 1:model.U, i in 1:model.K
		x += (model.e - 1) * (digamma(model.he[i,u]) - log(model.vav[i,u])) - model.f * model.he[i,u] / model.vav[i,u]
	end
	return x
end

function Elogpepsilon(model::CTPF, d::Int)
	x = model.K * (model.g * log(model.h) - lgamma(model.g))
	for i in 1:model.K
		x += (model.g - 1) * (digamma(model.zayin[d][i]) - log(model.het[d][i])) - model.h * model.zayin[d][i] / model.het[d][i]
	end
	return x
end

function Elogqy(model::CTPF, d::Int)
	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[d][:,u]))
	end
	return x
end

function Elogqz(model::CTPF, d::Int)
	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[d][:,n]))
	end
	return x
end

function Elogqbeta(model::CTPF)
	x = 0
	for j in 1:model.V, i in 1:model.K
		x -= entropy(Gamma(model.alef[i,j], 1 / model.bet[i,j]))
	end
	return x
end

function Elogqtheta(model::CTPF, d::Int)
	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.gimel[d][i], 1 / model.dalet[d][i]))
	end
	return x
end

function Elogqeta(model::CTPF)
	x = 0
	for u in 1:model.U, i in 1:model.K
		x -= entropy(Gamma(model.he[i,u], 1 / model.vav[i,u]))
	end
	return x
end	

function Elogqepsilon(model::CTPF, d::Int)
	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.zayin[d][i], 1 / model.het[d][i]))
	end
	return x
end

function updateELBO!(model::CTPF)
	model.elbo = Elogpbeta(model) + Elogpeta(model) - Elogqbeta(model) - Elogqeta(model)
	for d in 1:model.M
		model.elbo += (Elogpya(model, d)
					+ Elogpyb(model, d)
					+ Elogpz(model, d)
					+ Elogptheta(model, d)
					+ Elogpepsilon(model, d)
					- Elogqy(model, d)
					- Elogqz(model, d) 
					- Elogqtheta(model, d)
					- Elogqepsilon(model, d))
	end
	return model.elbo
end

function updateAlef!(model::CTPF)
	model.alef = fill(model.a, model.K, model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.alef[:,terms] += model.phi[d] .* counts'
	end
end

function updateBet!(model::CTPF)
	model.bet = fill(model.b, model.K, model.V)
	for d in 1:model.M
		model.bet .+= model.gimel[d] ./ model.dalet[d]
	end
	model.alefbet = vec(sum(model.alef ./ model.bet, 2))
end

function updateGimel!(model::CTPF, d::Int)	
	counts, ratings = model.corp[d].counts, model.corp[d].ratings
	model.gimel[d] = model.c + model.phi[d] * counts + model.xi[d][1:model.K,:] * ratings
end

function updateDalet!(model::CTPF, d::Int)
	model.dalet[d] = model.d + model.alefbet + model.hevav
end

function updateHe!(model::CTPF)
	model.he = fill(model.e, model.K, model.U)
	for d in 1:model.M
		readers, ratings = model.corp[d].readers, model.corp[d].ratings
		model.he[:,readers] += (model.xi[d][1:model.K,:] + model.xi[d][model.K+1:end,:]) .* ratings'
	end
end

function updateVav!(model::CTPF)
	model.vav = fill(model.f, model.K, model.U)
	for d in 1:model.M
		model.vav .+= model.gimel[d] ./ model.dalet[d] + model.zayin[d] ./ model.het[d]
	end
	model.hevav = vec(sum(model.he ./ model.vav, 2))
end

function updateZayin!(model::CTPF, d::Int)
	ratings = model.corp[d].ratings
	model.zayin[d] = model.g + model.xi[d][model.K+1:end,:] * ratings
end

function updateHet!(model::CTPF, d::Int)
	model.het[d] = model.h + model.hevav
end

function updatePhi!(model::CTPF, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = exp(digamma(model.gimel[d]) - log(model.dalet[d]) .+ (digamma(model.alef[:,terms]) - log(model.bet[:,terms])))
	model.phi[d] ./= sum(model.phi[d], 1)
end

function updateXi!(model::CTPF, d::Int)
	readers = model.corp[d].readers
	model.xi[d][1:model.K,:] = exp(digamma(model.gimel[d]) - log(model.dalet[d]) .+ (digamma(model.he[:,readers]) - log(model.vav[:,readers])))
	model.xi[d][model.K+1:end,:] = exp(digamma(model.zayin[d]) - log(model.het[d]) .+ (digamma(model.he[:,readers]) - log(model.vav[:,readers])))
	model.xi[d] ./= sum(model.xi[d], 1)
end

function train!(model::CTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, vtol]))
	@assert all(ispositive([iter, viter, chkelbo]))
	checkmodel(model)

	for k in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				oldhet = model.het[d]
				updatePhi!(model, d)
				updateXi!(model, d)
				updateGimel!(model, d)
				updateDalet!(model, d)
				updateZayin!(model, d)
				updateHet!(model, d)
				if norm(oldhet - model.het[d]) < vtol
					break
				end
			end
		end
		updateAlef!(model)
		updateBet!(model)
		updateHe!(model)
		updateVav!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	Ebeta = model.alef ./ model.bet
	model.topics = [reverse(sortperm(vec(Ebeta[i,:]))) for i in 1:model.K]

	Eeta = model.he ./ model.vav
	for d in 1:model.M
		Etheta = model.gimel[d] ./ model.dalet[d]
		Eepsilon = model.zayin[d] ./ model.het[d]
		model.scores[d,:] = sum(Eeta .* (Etheta + Eepsilon), 1)
	end

	model.drecs = Vector{Int}[]
	for d in 1:model.M
		nr = setdiff(keys(model.corp.users), model.corp[d].readers)
		push!(model.drecs, nr[reverse(sortperm(vec(model.scores[d,nr])))])
	end

	model.urecs = Vector{Int}[]
	for u in 1:model.U
		ur = filter(d -> !(u in model.corp[d].readers), collect(1:model.M))
		push!(model.urecs, ur[reverse(sortperm(model.scores[ur,u]))])
	end
	nothing
end

