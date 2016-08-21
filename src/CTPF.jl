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
	bet::Vector{Float64}
	gimel::VectorList{Float64}
	dalet::Vector{Float64}
	he::Matrix{Float64}
	vav::Vector{Float64}
	zayin::VectorList{Float64}
	het::Vector{Float64}
	phi::Matrix{Float64}
	xi::Matrix{Float64}
	newalef::Matrix{Float64}
	newhe::Matrix{Float64}
	elbo::Float64
	newelbo::Float64

	function CTPF(corp::Corpus, K::Integer, pmodel::Union{Void, BaseTopicModel}=nothing)
		@assert ispositive(K)
		@assert !isempty(corp)	
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		R = [length(doc.readers) for doc in corp]

		@assert ispositive(U)
		@assert isequal(collect(1:U), sort(collect(keys(corp.users))))
		libs = [Int[] for _ in 1:U]
		for u in 1:U, d in 1:M
			u in corp[d].readers && push!(libs[u], d)
		end

		a, b, c, d, e, f, g, h = fill(0.1, 8)

		if isa(pmodel, Union{AbstractLDA, AbstractCTM})
			@assert isequal(size(pmodel.beta), (K, V))
			alef = exp(pmodel.beta)
			topics = pmodel.topics		
		elseif isa(pmodel, Union{AbstractfLDA, AbstractfCTM})
			@assert isequal(size(pmodel.fbeta), (K, V))
			alef = exp(pmodel.fbeta)
			topics = pmodel.topics
		else
			alef = exp(rand(Dirichlet(V, 1.0), K)' - 0.5)
			topics = [collect(1:V) for _ in 1:K]
		end
		
		bet = ones(K)
		gimel = [ones(K) for _ in 1:M]
		dalet = ones(K)	
		he = ones(K, U)
		vav = ones(K)
		zayin = [ones(K) for _ in 1:M]
		het = ones(K)
		phi = ones(K, N[1]) / K
		xi = ones(2K, R[1]) / 2K

		model = new(K, M, V, U, N, C, R, copy(corp), topics, zeros(M, U), libs, Vector[], Vector[], a, b, c, d, e, f, g, h, alef, bet, gimel, dalet, he, vav, zayin, het, phi, xi)
		fixmodel!(model, check=false)

		for d in 1:M
			model.phi = ones(K, N[d]) / K
			model.xi = ones(2K, R[d]) / 2K
			updateNewELBO!(model, d)
		end
		model.phi = ones(K, N[1]) / K
		model.xi = ones(2K, R[1]) / 2K
		updateELBO!(model)
		return model
	end
end

function Elogpya(model::CTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[i,u])
		x += (ra * model.xi[i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.gimel[d][i] / model.dalet[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::CTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[model.K+i,u])
		x += (ra * model.xi[model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.zayin[d][i] / model.het[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::CTPF, d::Int)
	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[i,n])
		x += (c * model.phi[i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.alef[i,j]) - log(model.bet[i]))
				- (model.gimel[d][i] / model.dalet[i]) * (model.alef[i,j] / model.bet[i]) - sum([pdf(binom, z) * lgamma(z + 1) for z in 0:c]))
	end
	return x
end

function Elogpbeta(model::CTPF)
	x = model.V * model.K * (model.a * log(model.b) - lgamma(model.a))
	for j in 1:model.V, i in 1:model.K
		x += (model.a - 1) * (digamma(model.alef[i,j]) - log(model.bet[i])) - model.b * model.alef[i,j] / model.bet[i]
	end
	return x
end

function Elogptheta(model::CTPF, d::Int)
	x = model.K * (model.c * log(model.d) - lgamma(model.c))
	for i in 1:model.K
		x += (model.c - 1) * (digamma(model.gimel[d][i]) - log(model.dalet[i])) - model.d * model.gimel[d][i] / model.dalet[i]
	end
	return x
end

function Elogpeta(model::CTPF)
	x = model.U * model.K * (model.e * log(model.f) - lgamma(model.e))
	for u in 1:model.U, i in 1:model.K
		x += (model.e - 1) * (digamma(model.he[i,u]) - log(model.vav[i])) - model.f * model.he[i,u] / model.vav[i]
	end
	return x
end

function Elogpepsilon(model::CTPF, d::Int)
	x = model.K * (model.g * log(model.h) - lgamma(model.g))
	for i in 1:model.K
		x += (model.g - 1) * (digamma(model.zayin[d][i]) - log(model.het[i])) - model.h * model.zayin[d][i] / model.het[i]
	end
	return x
end

function Elogqy(model::CTPF, d::Int)
	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[:,u]))
	end
	return x
end

function Elogqz(model::CTPF, d::Int)
	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[:,n]))
	end
	return x
end

function Elogqbeta(model::CTPF)
	x = 0
	for j in 1:model.V, i in 1:model.K
		x -= entropy(Gamma(model.alef[i,j], 1 / model.bet[i]))
	end
	return x
end

function Elogqtheta(model::CTPF, d::Int)
	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.gimel[d][i], 1 / model.dalet[i]))
	end
	return x
end

function Elogqeta(model::CTPF)
	x = 0
	for u in 1:model.U, i in 1:model.K
		x -= entropy(Gamma(model.he[i,u], 1 / model.vav[i]))
	end
	return x
end	

function Elogqepsilon(model::CTPF, d::Int)
	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.zayin[d][i], 1 / model.het[i]))
	end
	return x
end

function updateELBO!(model::CTPF)
	model.elbo = model.newelbo + Elogpbeta(model) + Elogpeta(model) - Elogqbeta(model) - Elogqeta(model)
	model.newelbo = 0
	return model.elbo
end

function updateNewELBO!(model::CTPF, d::Int)
	model.newelbo += (Elogpya(model, d)
					+ Elogpyb(model, d)
					+ Elogpz(model, d)
					+ Elogptheta(model, d)
					+ Elogpepsilon(model, d)
					- Elogqy(model, d)
					- Elogqz(model, d) 
					- Elogqtheta(model, d)
					- Elogqepsilon(model, d))
end

function updateAlef!(model::CTPF)
	model.alef = copy(model.newalef)
	model.newalef = fill(model.a, model.K, model.V)
end

function updateNewAlef!(model::CTPF, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.newalef[:,terms] += model.phi .* counts'
end

function updateBet!(model::CTPF)
	model.bet = model.b + sum(model.gimel) ./ model.dalet
end

function updateGimel!(model::CTPF, d::Int)	
	counts, ratings = model.corp[d].counts, model.corp[d].ratings
	model.gimel[d] = model.c + model.phi * counts + model.xi[1:model.K,:] * ratings
end

function updateDalet!(model::CTPF)
	model.dalet = model.d + vec(sum(model.alef, 2)) ./ model.bet + vec(sum(model.he, 2)) ./ model.vav
end

function updateHe!(model::CTPF)
	model.he = copy(model.newhe)
	model.newhe = fill(model.e, model.K, model.U)
end

function updateNewHe!(model::CTPF, d::Int)
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	model.newhe[:,readers] += (model.xi[1:model.K,:] + model.xi[model.K+1:end,:]) .* ratings'
end

function updateVav!(model::CTPF)
	model.vav = model.f + sum(model.gimel) ./ model.dalet + sum(model.zayin) ./ model.het
end

function updateZayin!(model::CTPF, d::Int)
	ratings = model.corp[d].ratings
	model.zayin[d] = model.g + model.xi[model.K+1:end,:] * ratings
end

function updateHet!(model::CTPF)
	model.het = model.h + vec(sum(model.he, 2)) ./ model.vav
end

function updatePhi!(model::CTPF, d::Int)
	terms = model.corp[d].terms
	model.phi = exp(digamma(model.gimel[d]) - log(model.dalet) - log(model.bet) .+ digamma(model.alef[:,terms]))
	model.phi ./= sum(model.phi, 1)
end

function updateXi!(model::CTPF, d::Int)
	readers = model.corp[d].readers
	model.xi = vcat(exp(digamma(model.gimel[d]) - log(model.dalet) - log(model.vav) .+ digamma(model.he[:,readers])), 
					exp(digamma(model.zayin[d]) - log(model.het) - log(model.vav) .+ digamma(model.he[:,readers])))
	model.xi ./= sum(model.xi, 1)
end

function train!(model::CTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, vtol]))
	@assert all(ispositive([iter, viter, chkelbo]))

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for d in 1:model.M
			for _ in 1:viter
				oldgimel = model.gimel[d]
				updatePhi!(model, d)
				updateXi!(model, d)
				updateGimel!(model, d)
				updateZayin!(model, d)
				if norm(oldgimel - model.gimel[d]) < vtol
					break
				end
			end
			updateNewAlef!(model, d)
			updateNewHe!(model, d)
			chk && updateNewELBO!(model, d)
		end
		if checkELBO!(model, k, chk, tol)
			updateDalet!(model)
			updateHet!(model)
			updateAlef!(model)
			updateBet!(model)
			updateHe!(model)
			updateVav!(model)
			break
		end
		updateDalet!(model)
		updateHet!(model)
		updateAlef!(model)
		updateBet!(model)
		updateHe!(model)
		updateVav!(model)
	end
	updatePhi!(model, 1)
	updateXi!(model, 1)
	
	Ebeta = model.alef ./ model.bet
	model.topics = [reverse(sortperm(vec(Ebeta[i,:]))) for i in 1:model.K]

	Eeta = model.he ./ model.vav
	for d in 1:model.M
		Etheta = model.gimel[d] ./ model.dalet
		Eepsilon = model.zayin[d] ./ model.het
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

