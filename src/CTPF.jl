mutable struct CTPF <: TopicModel
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
	alef_old::Matrix{Float64}
	alef_temp::Matrix{Float64}
	he::Matrix{Float64}
	he_old::Matrix{Float64}
	he_temp::Matrix{Float64}
	bet::Vector{Float64}
	bet_old::Vector{Float64}
	vav::Vector{Float64}
	vav_old::Vector{Float64}
	gimel::VectorList{Float64}
	gimel_old::VectorList{Float64}
	zayin::VectorList{Float64}
	zayin_old::VectorList{Float64}
	dalet::Vector{Float64}
	dalet_old::Vector{Float64}
	het::Vector{Float64}
	het_old::Vector{Float64}
	phi::MatrixList{Float64}
	xi::MatrixList{Float64}
	elbo::Float64

	function CTPF(corp::Corpus, K::Integer)
		check_corp(corp)
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		R = [length(doc.readers) for doc in corp]

		topics = [collect(1:V) for _ in 1:K]
		scores = zeros(M, U)

		libs = [Int[] for _ in 1:U]
		for u in 1:U, d in 1:M
			u in corp[d].readers && push!(libs[u], d)
		end

		drecs = Vector[]
		urecs = Vector[]

		a, b, c, d, e, f, g, h = fill(0.1, 8)

		alef = exp.(rand(Dirichlet(V, 1.0), K)' .- 0.5)
		alef_old = copy(alef)
		alef_temp = fill(a, K, V)
		he = ones(K, U)
		he_old = copy(he)
		he_temp = fill(e, K, U)
		bet = ones(K)
		bet_old = copy(bet)
		vav = ones(K)
		vav_old = copy(vav)
		gimel = [ones(K) for _ in 1:M]
		gimel_old = copy(gimel)
		zayin = [ones(K) for _ in 1:M]
		zayin_old = copy(zayin)
		dalet = ones(K)
		dalet_old = copy(dalet)
		het = ones(K)
		het_old = copy(het)
		phi = [ones(K, N[d]) / K for d in 1:min(M, 1)]
		xi = [ones(2K, R[d]) / 2K for d in 1:min(M, 1)]
		elbo = 0

		model = new(K, M, V, U, N, C, R, copy(corp), topics, scores, libs, drecs, urecs, a, b, c, d, e, f, g, h, alef, alef_old, alef_temp, he, he_old, he_temp, bet, bet_old, vav, vav_old, gimel, gimel_old, zayin, zayin_old, dalet, dalet_old, het, het_old, phi, xi, elbo)

		for d in 1:model.M
			model.phi[1] = ones(K, N[d]) / K
			model.xi[1] = ones(2K, R[d]) / 2K
			model.elbo += Elogpya(model, d) + Elogpyb(model, d) + Elogpz(model, d) + Elogptheta(model, d) + Elogpepsilon(model, d) - Elogqy(model, d) - Elogqz(model, d) - Elogqtheta(model, d) - Elogqepsilon(model, d)
		end

		return model
	end
end

function Elogpya(model::CTPF, d::Int)
	"Compute E[log(P(ya))]."

	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[1][i,u])
		x += (ra * model.xi[1][i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.he[i,re]) - log(model.vav[i])) - (model.gimel[d][i] / model.dalet[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * loggamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::CTPF, d::Int)
	"Compute E[log(P(yb))]."

	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[1][model.K+i,u])
		x += (ra * model.xi[1][model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[i]) + digamma(model.he[i,re]) - log(model.vav[i])) - (model.zayin[d][i] / model.het[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * loggamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::CTPF, d::Int)
	"Compute E[log(P(z))]."

	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[1][i,n])
		x += (c * model.phi[1][i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.alef[i,j]) - log(model.bet[i])) - (model.gimel[d][i] / model.dalet[i]) * (model.alef[i,j] / model.bet[i]) - sum([pdf(binom, z) * loggamma(z + 1) for z in 0:c]))
	end
	return x
end

function Elogpbeta(model::CTPF)
	"Compute E[log(P(beta))]."

	x = model.V * model.K * (model.a * log(model.b) -  loggamma(model.a))
	for j in 1:model.V, i in 1:model.K
		x += (model.a - 1) * (digamma(model.alef[i,j]) - log(model.bet[i])) - model.b * model.alef[i,j] / model.bet[i]
	end
	return x
end

function Elogptheta(model::CTPF, d::Int)
	"Compute E[log(P(theta))]."

	x = model.K * (model.c * log(model.d) -  loggamma(model.c))
	for i in 1:model.K
		x += (model.c - 1) * (digamma(model.gimel[d][i]) - log(model.dalet[i])) - model.d * model.gimel[d][i] / model.dalet[i]
	end
	return x
end

function Elogpeta(model::CTPF)
	"Compute E[log(P(eta))]."

	x = model.U * model.K * (model.e * log(model.f) -  loggamma(model.e))
	for u in 1:model.U, i in 1:model.K
		x += (model.e - 1) * (digamma(model.he[i,u]) - log(model.vav[i])) - model.f * model.he[i,u] / model.vav[i]
	end
	return x
end

function Elogpepsilon(model::CTPF, d::Int)
	"Compute E[log(P(epsilon))]."

	x = model.K * (model.g * log(model.h) -  loggamma(model.g))
	for i in 1:model.K
		x += (model.g - 1) * (digamma(model.zayin[d][i]) - log(model.het[i])) - model.h * model.zayin[d][i] / model.het[i]
	end
	return x
end

function Elogqy(model::CTPF, d::Int)
	"Compute E[log(q(y))]."

	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[1][:,u]))
	end
	return x
end

function Elogqz(model::CTPF, d::Int)
	"Compute E[log(q(z))]."

	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[1][:,n]))
	end
	return x
end

function Elogqbeta(model::CTPF)
	"Compute E[log(q(beta))]."

	x = 0
	for j in 1:model.V, i in 1:model.K
		x -= entropy(Gamma(model.alef[i,j], 1 / model.bet[i]))
	end
	return x
end

function Elogqtheta(model::CTPF, d::Int)
	"Compute E[log(q(theta))]."

	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.gimel[d][i], 1 / model.dalet[i]))
	end
	return x
end

function Elogqeta(model::CTPF)
	"Compute E[log(q(eta))]."

	x = 0
	for u in 1:model.U, i in 1:model.K
		x -= entropy(Gamma(model.he[i,u], 1 / model.vav[i]))
	end
	return x
end	

function Elogqepsilon(model::CTPF, d::Int)
	"Compute E[log(q(epsilon))]."

	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.zayin[d][i], 1 / model.het[i]))
	end
	return x
end

function update_elbo!(model::CTPF)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		terms = model.corp[d].terms
		readers = model.corp[d].readers

		model.phi[1] = exp.(digamma.(model.gimel_old[d]) - log.(model.dalet_old) - log.(model.bet_old) .+ digamma.(model.alef_old[:,terms]))
		model.phi[1] ./= sum(model.phi[1], dims=1)

		model.xi[1] = vcat(exp.(digamma.(model.gimel_old[d]) - log.(model.dalet_old) - log.(model.vav_old) .+ digamma.(model.he_old[:,readers])), exp.(digamma.(model.zayin_old[d]) - log.(model.het_old) - log.(model.vav_old) .+ digamma.(model.he_old[:,readers])))
		model.xi[1] ./= sum(model.xi[1], dims=1)
		
		model.elbo += Elogpya(model, d) + Elogpyb(model, d) + Elogpz(model, d) + Elogptheta(model, d) + Elogpepsilon(model, d) - Elogqy(model, d) - Elogqz(model, d) - Elogqtheta(model, d) - Elogqepsilon(model, d)
	end

	return model.elbo
end

function update_alef!(model::CTPF)
	"Update alef."
	"Analytic."

	model.alef_old = model.alef
	model.alef = model.alef_temp
	model.alef_temp = fill(model.a, model.K, model.V)
end

function update_alef!(model::CTPF, d::Int)
	"Update alef."
	"Analytic."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.alef_temp[:,terms] += model.phi[1] .* counts'
end

function update_he!(model::CTPF)
	"Update he."
	"Analytic."

	model.he_old = copy(model.he)
	model.he = model.he_temp
	model.he_temp = fill(model.e, model.K, model.U)
end

function update_he!(model::CTPF, d::Int)
	"Update he."
	"Analytic."

	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	model.he_temp[:,readers] += (model.xi[1][1:model.K,:] + model.xi[1][model.K+1:end,:]) .* ratings'
end

function update_bet!(model::CTPF)
	"Update bet."
	"Analytic"

	model.bet_old = model.bet
	model.bet = model.b .+ sum(model.gimel) ./ model.dalet
end

function update_vav!(model::CTPF)
	"Update vav."
	"Analytic."

	model.vav_old = model.vav
	model.vav = model.f .+ sum(model.gimel) ./ model.dalet + sum(model.zayin) ./ model.het
end

function update_gimel!(model::CTPF, d::Int)
	"Update gimel."
	"Analytic."

	model.gimel_old[d] = model.gimel[d]

	counts, ratings = model.corp[d].counts, model.corp[d].ratings
	model.gimel[d] = model.c .+ model.phi[1] * counts + model.xi[1][1:model.K,:] * ratings
end

function update_zayin!(model::CTPF, d::Int)
	"Update zayin."
	"Analytic"

	model.zayin_old[d] = model.zayin[d]

	ratings = model.corp[d].ratings
	model.zayin[d] = model.g .+ model.xi[1][model.K+1:end,:] * ratings
end

function update_dalet!(model::CTPF)
	"Update dalet."
	"Analytic."

	model.dalet_old = model.dalet
	model.dalet = model.d .+ vec(sum(model.alef, dims=2)) ./ model.bet + vec(sum(model.he, dims=2)) ./ model.vav
end

function update_het!(model::CTPF)
	"Update het."
	"Analytic"

	model.het_old = model.het
	model.het = model.h .+ vec(sum(model.he, dims=2)) ./ model.vav
end

function update_phi!(model::CTPF, d::Int)
	"Update phi."
	"Analytic."

	terms = model.corp[d].terms
	model.phi[1] = exp.(digamma.(model.gimel[d]) - log.(model.dalet) - log.(model.bet) .+ digamma.(model.alef[:,terms]))
	model.phi[1] ./= sum(model.phi[1], dims=1)
end

function update_xi!(model::CTPF, d::Int)
	"Update xi."
	"Analytic."

	readers = model.corp[d].readers
	model.xi[1] = vcat(exp.(digamma.(model.gimel[d]) - log.(model.dalet) - log.(model.vav) .+ digamma.(model.he[:,readers])), exp.(digamma.(model.zayin[d]) - log.(model.het) - log.(model.vav) .+ digamma.(model.he[:,readers])))
	model.xi[1] ./= sum(model.xi[1], dims=1)
end

function train!(model::CTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for collaborative topic Poisson factorization variational Bayes algorithm."

	check_model(model)
	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .> 0)										|| throw(ArgumentError("Iteration parameters must be positive integers."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf) || throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))
	all([isempty(doc) for doc in corp]) && (iter = 0)
	update_elbo!(model)

	for k in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				update_xi!(model, d)
				update_phi!(model, d)
				update_zayin!(model, d)
				update_gimel!(model, d)
				if norm(model.gimel[d] - model.gimel_old[d]) < vtol
					break
				end
			end
			update_alef!(model, d)
			update_he!(model, d)
		end
		update_dalet!(model)
		update_het!(model)
		update_alef!(model)
		update_bet!(model)
		update_he!(model)
		update_vav!(model)

		if check_elbo!(model, check_elbo, k, tol)
			break
		end
	end
	return nothing
	
	Ebeta = model.alef ./ model.bet
	model.topics = [reverse(sortperm(vec(Ebeta[i,:]))) for i in 1:model.K]

	Eeta = model.he ./ model.vav
	for d in 1:model.M
		Etheta = model.gimel[d] ./ model.dalet
		Eepsilon = model.zayin[d] ./ model.het
		model.scores[d,:] = sum(Eeta .* (Etheta + Eepsilon), dims=1)
	end

	model.drecs = Vector{Int}[]
	for d in 1:model.M
		nr = collect(setdiff(keys(model.corp.users), model.corp[d].readers))
		push!(model.drecs, nr[reverse(sortperm(vec(model.scores[d,nr])))])
	end

	model.urecs = Vector{Int}[]
	for u in 1:model.U
		ur = filter(d -> !(u in model.corp[d].readers), collect(1:model.M))
		push!(model.urecs, ur[reverse(sortperm(model.scores[ur,u]))])
	end
	nothing
end

Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")