function check_model(model::gpuCTPF)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])
	@assert isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])
	@assert isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])
	@assert ispositive(model.a)
	@assert ispositive(model.b)
	@assert ispositive(model.c)
	@assert ispositive(model.d)
	@assert ispositive(model.e)
	@assert ispositive(model.f)
	@assert ispositive(model.g)
	@assert ispositive(model.h)	
	@assert isequal(size(model.alef), (model.K, model.V))
	@assert all(isfinite.(model.alef))
	@assert all(ispositive.(model.alef))
	@assert isequal(length(model.bet), model.K)
	@assert all(isfinite.(model.bet))
	@assert all(ispositive.(model.bet))
	@assert isequal(length(model.gimel), model.M)
	@assert all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.gimel[d])) for d in 1:model.M])
	@assert isequal(length(model.dalet), model.K)
	@assert all(isfinite.(model.dalet))
	@assert all(ispositive.(model.dalet))
	@assert isequal(size(model.he), (model.K, model.U))	
	@assert all(isfinite.(model.he))
	@assert all(ispositive.(model.he))
	@assert isequal(length(model.vav), model.K)
	@assert all(isfinite.(model.vav))
	@assert all(ispositive.(model.vav))
	@assert isequal(length(model.zayin), model.M)
	@assert all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.zayin[d])) for d in 1:model.M])
	@assert isequal(length(model.het), model.K)
	@assert all(isfinite.(model.het))
	@assert all(ispositive.(model.het))
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])
	@assert isequal(length(model.xi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in model.batches[1]])

	model.newalef = nothing
	model.newhe = nothing
		
	model.terms = [vcat([doc.terms for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.counts = [vcat([doc.counts for doc in model.corp[batch]]...) for batch in model.batches]
	model.words = [sortperm(termvec) - 1 for termvec in model.terms]

	model.readers = [vcat([doc.readers for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.ratings = [vcat([doc.ratings for doc in model.corp[batch]]...) for batch in model.batches]
	model.views = [sortperm(readervec) - 1 for readervec in model.readers]

	model.Npsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	model.Rpsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	for (b, batch) in enumerate(model.batches)
		for (m, d) in enumerate(batch)
			model.Npsums[b][m+1] = model.Npsums[b][m] + model.N[d]
			model.Rpsums[b][m+1] = model.Rpsums[b][m] + model.R[d]
		end
	end
		
	J = [zeros(Int, model.V) for _ in 1:model.B]
	for in 1:model.B
		for j in model.terms[b]
			J[b][j+1] += 1
		end
	end

	model.Jpsums = [zeros(Int, model.V + 1) for _ in 1:model.B]
	for in 1:model.B
		for j in 1:model.V
			model.Jpsums[b][j+1] = model.Jpsums[b][j] + J[b][j]
		end
	end

	Y = [zeros(Int, model.U) for _ in 1:model.B]
	for in 1:model.B
		for r in model.readers[b]
			Y[b][r+1] += 1
		end
	end

	model.Ypsums = [zeros(Int, model.U + 1) for _ in 1:model.B]
	for in 1:model.B
		for u in 1:model.U
			model.Ypsums[b][u+1] = model.Ypsums[b][u] + Y[b][u]
		end
	end


		
	@buf model.alef
	@buf model.bet
	@buf model.gimel
	@buf model.dalet
	@buf model.he
	@buf model.vav
	@buf model.zayin
	@buf model.het
	@buf model.newalef
	@buf model.newhe
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end

mutable struct gpuCTPF <: TopicModel
	K::Int
	M::Int
	V::Int
	U::Int
	N::Vector{Int}
	C::Vector{Int}
	R::Vector{Int}
	B::Int
	corp::Corpus
	batches::Vector{UnitRange{Int}}
	topics::VectorList{Int}
	scores::Matrix{Float32}
	libs::VectorList{Int}
	drecs::VectorList{Int}
	urecs::VectorList{Int}
	a::Float32
	b::Float32
	c::Float32
	d::Float32
	e::Float32
	f::Float32
	g::Float32
	h::Float32
	alef::Matrix{Float32}
	bet::Vector{Float32}
	gimel::VectorList{Float32}
	dalet::Vector{Float32}
	he::Matrix{Float32}
	vav::Vector{Float32}
	zayin::VectorList{Float32}
	het::Vector{Float32}
	phi::MatrixList{Float32}
	xi::MatrixList{Float32}
	elbo::Float32
	Npsums::VectorList{Int}
	Jpsums::VectorList{Int}
	Rpsums::VectorList{Int}
	Ypsums::VectorList{Int}
	terms::VectorList{Int}
	counts::VectorList{Int}
	words::VectorList{Int}
	readers::VectorList{Int}
	ratings::VectorList{Int}
	views::VectorList{Int}
	device::cl.Device
	context::cl.Context
	queue::cl.CmdQueue
	N_partial_sums_buffer::cl.Buffer{Int}
	J_partial_sums_buffer::cl.Buffer{Int}
	R_partial_sums_buffer::cl.Buffer{Int}
	Ypsumsbuf::cl.Buffer{Int}
	termsbuf::cl.Buffer{Int}
	counts_buffer::cl.Buffer{Int}
	terms_sortperm_buffer::cl.Buffer{Int}
	readers_buffer::cl.Buffer{Int}
	ratingsbuf::cl.Buffer{Int}
	viewsbuf::cl.Buffer{Int}
	alef_kernel::cl.Kernel
	bet_kernel::cl.Kernel
	gimelkern::cl.Kernel
	daletkern::cl.Kernel
	hekern::cl.Kernel
	vav_kernel::cl.Kernel
	zayin_kernel::cl.Kernel
	het_kernel::cl.Kernel
	phi_kernel::cl.Kernel
	phi_norm_kernel::cl.Kernel
	xi_kernel::cl.Kernel
	xi_norm_kernel::cl.Kernel
	alef_buffer::cl.Buffer{Float32}
	bet_buffer::cl.Buffer{Float32}
	gimel_buffer::cl.Buffer{Float32}
	dalet_buffer::cl.Buffer{Float32}
	hebuf::cl.Buffer{Float32}
	vav_buffer::cl.Buffer{Float32}
	zayin_buffer::cl.Buffer{Float32}
	het_buffer::cl.Buffer{Float32}
	phi_buffer::cl.Buffer{Float32}
	xi_buffer::cl.Buffer{Float32}

	function gpuCTPF(corp::Corpus, K::Integer)
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

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

		if isa(basemodel, Union{AbstractLDA, AbstractCTM})
			@assert isequal(size(basemodel.beta), (K, V))
			alef = exp.(basemodel.beta)
			topics = basemodel.topics		
		elseif isa(basemodel, Union{AbstractfLDA, AbstractfCTM})
			@assert isequal(size(basemodel.fbeta), (K, V))
			alef = exp.(basemodel.fbeta)
			topics = basemodel.topics
		else
			alef = exp.(rand(Dirichlet(V, 1.0), K)' - 0.5)
			topics = [collect(1:V) for _ in 1:K]
		end
		
		bet = ones(K)
		gimel = [ones(K) for _ in 1:M]
		dalet = ones(K)
		he = ones(K, U)
		vav = ones(K)
		zayin = [ones(K) for _ in 1:M]
		het = ones(K)
		phi = [ones(K, N[d]) / K for d in 1:M]
		xi = [ones(2K, R[d]) / 2K for d in 1:M]
		elbo = 0

		device, context, queue = cl.create_compute_context()		

		alef_prog = cl.Program(context, source=CTPF_ALEF_c) |> cl.build!
		bet_program = cl.Program(context, source=CTPF_BET_c) |> cl.build!
		gimel_program = cl.Program(context, source=CTPF_GIMEL_c) |> cl.build!
		dalet_program = cl.Program(context, source=CTPF_DALET_c) |> cl.build!
		he_program = cl.Program(context, source=CTPF_HE_c) |> cl.build!
		vav_program = cl.Program(context, source=CTPF_VAV_c) |> cl.build!
		zayin_program = cl.Program(context, source=CTPF_ZAYIN_c) |> cl.build!
		het_program = cl.Program(context, source=CTPF_HET_c) |> cl.build!
		phi_program = cl.Program(context, source=CTPF_PHI_c) |> cl.build!
		phi_norm_program = cl.Program(context, source=CTPF_PHI_NORM_c) |> cl.build!
		xi_program = cl.Program(context, source=CTPF_XI_c) |> cl.build!
		xi_norm_program = cl.Program(context, source=CTPF_XI_NORM_c) |> cl.build!

		alef_kernel = cl.Kernel(alef_program, "update_alef")
		bet_kernel = cl.Kernel(bet_program, "update_bet")
		gimel_kernel = cl.Kernel(gimel_program, "update_gimel")
		dalet_kernel = cl.Kernel(dalet_program, "update_dalet")
		he_kernel = cl.Kernel(he_program, "update_he")
		vav_kernel = cl.Kernel(vav_program, "update_vav")
		zayin_kernel = cl.Kernel(zayin_program, "update_zayin")
		het_kernel = cl.Kernel(het_program, "update_het")
		phi_kernel = cl.Kernel(phi_program, "update_phi")
		phi_norm_kernel = cl.Kernel(phi_norm_program, "normalize_phi")
		xi_kernel = cl.Kernel(xi_program, "update_xi")
		xi_norm_kernel = cl.Kernel(xi_norm_program, "normalize_xi")

		model = new(K, M, V, U, N, C, R, copy(corp), topics, scores, libs, drecs, urecs, a, b, c, d, e, f, g, h, alef, alef_old, alef_temp, he, he_old, he_temp, bet, bet_old, vav, vav_old, gimel, gimel_old, zayin, zayin_old, dalet, dalet_old, het, het_old, phi, xi, elbo, alef_kernel, bet_kernel, gimel_kernel, dalet_kernel, he_kernel, vav_kernel, zayin_kernel, het_kernel, phi_kernel, phi_norm_kernel, xi_kernel, xi_norm_kernel)
		update_elbo!(model)
		return model
	end
end

function Elogpya(model::gpuCTPF, d::Int)
	"Compute E[log(P(ya))]."

	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[i,u])
		x += (ra * model.xi[i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.he[i,re]) - log(model.vav[i])) - (model.gimel[d][i] / model.dalet[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * loggamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::gpuCTPF, d::Int)
	"Compute E[log(P(yb))]."

	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[model.K+i,u])
		x += (ra * model.xi[model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[i]) + digamma(model.he[i,re]) - log(model.vav[i])) - (model.zayin[d][i] / model.het[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * loggamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::gpuCTPF, d::Int)
	"Compute E[log(P(z))]."

	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[i,n])
		x += (c * model.phi[i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.alef[i,j]) - log(model.bet[i])) - (model.gimel[d][i] / model.dalet[i]) * (model.alef[i,j] / model.bet[i]) - sum([pdf(binom, z) * loggamma(z + 1) for z in 0:c]))
	end
	return x
end

function Elogpbeta(model::gpuCTPF)
	"Compute E[log(P(beta))]."

	x = model.V * model.K * (model.a * log(model.b) -  loggamma(model.a))
	for j in 1:model.V, i in 1:model.K
		x += (model.a - 1) * (digamma(model.alef[i,j]) - log(model.bet[i])) - model.b * model.alef[i,j] / model.bet[i]
	end
	return x
end

function Elogptheta(model::gpuCTPF, d::Int)
	"Compute E[log(P(theta))]."

	x = model.K * (model.c * log(model.d) -  loggamma(model.c))
	for i in 1:model.K
		x += (model.c - 1) * (digamma(model.gimel[d][i]) - log(model.dalet[i])) - model.d * model.gimel[d][i] / model.dalet[i]
	end
	return x
end

function Elogpeta(model::gpuCTPF)
	"Compute E[log(P(eta))]."

	x = model.U * model.K * (model.e * log(model.f) -  loggamma(model.e))
	for u in 1:model.U, i in 1:model.K
		x += (model.e - 1) * (digamma(model.he[i,u]) - log(model.vav[i])) - model.f * model.he[i,u] / model.vav[i]
	end
	return x
end

function Elogpepsilon(model::gpuCTPF, d::Int)
	"Compute E[log(P(epsilon))]."

	x = model.K * (model.g * log(model.h) -  loggamma(model.g))
	for i in 1:model.K
		x += (model.g - 1) * (digamma(model.zayin[d][i]) - log(model.het[i])) - model.h * model.zayin[d][i] / model.het[i]
	end
	return x
end

function Elogqy(model::gpuCTPF, d::Int)
	"Compute E[log(q(y))]."

	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[:,u]))
	end
	return x
end

function Elogqz(model::gpuCTPF, d::Int)
	"Compute E[log(q(z))]."

	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[:,n]))
	end
	return x
end

function Elogqbeta(model::gpuCTPF)
	"Compute E[log(q(beta))]."

	x = 0
	for j in 1:model.V, i in 1:model.K
		x -= entropy(Gamma(model.alef[i,j], 1 / model.bet[i]))
	end
	return x
end

function Elogqtheta(model::gpuCTPF, d::Int)
	"Compute E[log(q(theta))]."

	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.gimel[d][i], 1 / model.dalet[i]))
	end
	return x
end

function Elogqeta(model::gpuCTPF)
	"Compute E[log(q(eta))]."

	x = 0
	for u in 1:model.U, i in 1:model.K
		x -= entropy(Gamma(model.he[i,u], 1 / model.vav[i]))
	end
	return x
end	

function Elogqepsilon(model::gpuCTPF, d::Int)
	"Compute E[log(q(epsilon))]."

	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.zayin[d][i], 1 / model.het[i]))
	end
	return x
end

function updateELBO!(model::gpuCTPF)
	model.elbo = model.newelbo + Elogpbeta(model) + Elogpeta(model) - Elogqbeta(model) - Elogqeta(model)
	model.newelbo = 0
	return model.elbo
end

function update_elbo!(model::gpuCTPF)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		model.elbo += Elogpya(model, d) + Elogpyb(model, d) + Elogpz(model, d) + Elogptheta(model, d) + Elogpepsilon(model, d) - Elogqy(model, d) - Elogqz(model, d) - Elogqtheta(model, d) - Elogqepsilon(model, d)
	end
end

const CTPF_ALEF_c =
"""
kernel void
update_alef(long K,
			const global long *J_partial_sums,
			const global long *terms_sortperm,
			const global long *counts,

			const global float *phi,
			global float *alef)
						
			{
			long i = get_global_id(0);
			long j = get_global_id(1);	

			float acc = 0.0f;

			for (long w=J_partial_sums[j]; w<J_partial_sums[j+1]; w++)
				acc += counts[terms_sortperm[w]] * phi[K * terms_sortperm[w] + i];

			alef[K * j + i] = acc;
			}
			"""

function update_alef!(model::gpuCTPF)
	model.queue(model.alef_kernel, (model.K, model.V), nothing, model.K, model.J_partial_sums_buffer, model.counts_buffer, model.terms_sortperm_buffer, model.phi_buffer, model.alef_buffer)
end

const CTPF_BET_c = 
"""
kernel void
update_bet(	long K,
			long M,
			float b,
			const global float *alef,
			const global float *gimel,
			const global float *dalet,
			global float *bet)

			{
			long i = get_global_id(0);

			float acc = 0.0f;

			for (long d=0; d<M; d++)
				acc += gimel[K * d + i];

			bet[i] = b + acc / dalet[i];
			}
			"""

function update_bet!(model::gpuCTPF)
	model.queue(model.bet_kernel, model.K, nothing, model.K, model.M, model.b, model.alef_buffer, model.gimel_buffer, model.dalet_buffer, model.bet_buffer)
end

const CTPF_GIMEL_c = 
"""
kernel void
updateGimel(long K,
			float c,
			const global long *Npsums,
			const global long *Rpsums,
			const global long *counts,
			const global long *ratings,
			const global float *phi,
			const global float *xi,
			global float *gimel)

			{   
			long i = get_global_id(0);
			long d = get_global_id(1);

			float acc_phi = 0.0f;
			float acc_xi = 0.0f;

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				acc_phi += phi[K * n + i] * counts[n];

			for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
				acc_xi += xi[2 * K * r + i] * ratings[r]; 

			gimel[K * d + i] = c + acc_phi + acc_xi;
			}
			"""

function update_gimel!(model::gpuCTPF, b::Int)
	"Update gimel."
	"Analytic."

	model.gimel_old[d] = model.gimel[d]

	model.queue(model.gimelkern, (model.K, model.M), nothing, model.K, model.c, model.N_partial_sums_buffer, model.R_partial_sums_buffer, model.counts_buffer, model.ratingsbuf, model.phi_buffer, model.xi_buffer, model.gimel_buffer)
	@buffer model.gimel
end

const CTPF_DALET_c =
"""
kernel void
updateDalet(long K,
			long V,
			long U,
			float d,
			const global float *alef,
			const global float *bet,
			const global float *he,
			const global float *vav,
			global float *dalet)
			
			{
			long i = get_global_id(0);
			
			float accalef = 0.0f;
			float acche = 0.0f;
				
			for (long j=0; j<V; j++)
				accalef += alef[K * j + i];

			for (long u=0; u<U; u++)
				acche += he[K * u + i];

			dalet[i] = d + accalef / bet[i] + acche / vav[i];
			}
			"""

function update_dalet!(model::gpuCTPF)
	model.queue(model.daletkern, model.K, nothing, model.K, model.V, model.U, model.d, model.alef_buffer, model.bet_buffer, model.hebuf, model.vav_buffer, model.dalet_buffer)
end

function update_he!(model::gpuCTPF)
	model.queue(model.hekern, (model.K, model.U), nothing, model.K, model.e, model.newhebuf, model.hebuf)
end

const CTPF_HE_c =
"""
kernel void
update_he(long K,
			const global long *Ypsums,
			const global long *ratings,
			const global long *views,
			const global float *xi,
			global float *he)

			{
			long i = get_global_id(0);
			long u = get_global_id(1);

			float acc = 0.0f;

			for (long r=Ypsums[u]; r<Ypsums[u+1]; r++)
				acc += ratings[views[r]] * (xi[2 * K * views[r] + i] + xi[K * (2 * views[r] + 1) + i]);

			newhe[K * u + i] += acc;
			}
			"""

function udpate_he!(model::gpuCTPF)
	model.queue(model.he_kernel, (model.K, model.U), nothing, model.K, model.Ypsumsbuf, model.ratingsbuf, model.viewsbuf, model.xi_buffer, model.he_buffer)
end

const CTPF_VAV_c = 
"""
kernel void
update_vav(long K,
			long M,
			float f,
			const global float *gimel,
			const global float *dalet,
			const global float *zayin,
			const global float *het,
			global float *vav)

			{
			long i = get_global_id(0);

			float accgimel = 0.0f;
			float acczayin = 0.0f;

			for (long d=0; d<M; d++)
			{
				accgimel += gimel[K * d + i];
				acczayin += zayin[K * d + i];				
			}

			vav[i] = f + accgimel / dalet[i] + acczayin / het[i];
			}
			"""

function updateVav!(model::gpuCTPF)
	model.queue(model.vav_kernel, model.K, nothing, model.K, model.M, model.f, model.gimel_buffer, model.dalet_buffer, model.zayin_buffer, model.het_buffer, model.vav_buffer)
end

const CTPF_ZAYIN_c =
"""
kernel void
update_zayin(	long K,
				float g,
				const global long *Rpsums,
				const global long *ratings,
				const global float *xi,
				global float *zayin)

				{
				long i = get_global_id(0);
				long d = get_global_id(1);

				float acc = 0.0f;

				for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
					acc += xi[K * (2 * r + 1) + i] * ratings[r];

				zayin[K * (F + d) + i] = g + acc; 
				}
				"""

function update_zayin!(model::gpuCTPF)
	model.queue(model.zayin_kernel, (model.K, model.M), nothing, model.K, model.g, model.R_partial_sums_buffer, model.ratingsbuf, model.xi_buffer, model.zayin_buffer)
end

const CTPF_HET_c =
"""
kernel void
update_het(long K,
			long U,
			float h,
			const global float *he,
			const global float *vav,
			global float *het)

			{
			long i = get_global_id(0);

			float acc = 0.0f;

			for (long u=0; u<U; u++)
				acc += he[K * u + i];

			het[i] = h + acc / vav[i];
			}
			"""

function update_het!(model::gpuCTPF)
	model.queue(model.het_kernel, model.K, nothing, model.K, model.U, model.h, model.hebuf, model.vav_buffer, model.het_buffer)
end

const CTPF_PHI_c =
"""
$(DIGAMMA_c)

kernel void
update_phi(	long K,
			const global long *Npsums,
			const global long *terms,
			const global float *alef,
			const global float *bet,
			const global float *gimel,
			const global float *dalet,
			global float *phi)

			{

			long i = get_global_id(0);
			long d = get_global_id(1);

			float gdb = digamma(gimel[K * (F + d) + i]) - log(dalet[i]) - log(bet[i]);

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				phi[K * n + i] = exp(gdb + digamma(alef[K * terms[n] + i]));
			}
			"""

const CTPF_PHI_NORM_c =
"""
kernel void
normalize_phi(long K,
				global float *phi)
				
				{
				long dn = get_global_id(0);

				float normalizer = 0.0f;
											
				for (long i=0; i<K; i++)
					normalizer += phi[K * dn + i];

				for (long i=0; i<K; i++)
					phi[K * dn + i] /= normalizer;
				}
				"""

function updatePhi!(model::gpuCTPF, b::Int)
	batch = model.batches[b]
	model.queue(model.phi_kernel, (model.K, model.M), nothing, model.K, model.N_partial_sums_buffer, model.termsbuf, model.alef_buffer, model.bet_buffer, model.gimel_buffer, model.dalet_buffer, model.phi_buffer)
	model.queue(model.phi_norm_kernel, sum(model.N), nothing, model.K, model.phi_buffer)
end

const CTPF_XI_c =
"""
$(DIGAMMA_c)

kernel void
update_xi(long F,
			long K,
			const global long *Rpsums,
			const global long *readers,
			const global float *bet,
			const global float *gimel,
			const global float *dalet,
			const global float *he,
			const global float *vav,
			const global float *zayin,
			const global float *het,
			global float *xi)

			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			float gdv = digamma(gimel[K * (F + d) + i]) - log(dalet[i]) - log(bet[i]);
			float zhv = digamma(zayin[K * (F + d) + i]) - log(het[i]) - log(vav[i]);

			for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
			{
				xi[2 * K * r + i] = exp(gdv + digamma(he[K * readers[r] + i]));
				xi[K * (2 * r + 1) + i] = exp(zhv + digamma(he[K * readers[r] + i]));			
			}
			}
			"""

const CTPF_XI_NORM_c =
"""
kernel void
normalize_xi(long K,
			global float *xi)
				
				{
				long dn = get_global_id(0);

				float normalizer = 0.0f;
											
				for (long i=0; i<2*K; i++)
					normalizer += xi[2 * K * dn + i];

				for (long i=0; i<2*K; i++)
					xi[2 * K * dn + i] /= normalizer;
				}
				"""

function updateXi!(model::gpuCTPF, b::Int)
	batch = model.batches[b]
	model.queue(model.xi_kernel, (model.K, model.M), nothing, model.K, model.R_partial_sums_buffer, model.readers_buffer, model.bet_buffer, model.gimel_buffer, model.dalet_buffer, model.hebuf, model.vav_buffer, model.zayin_buffer, model.het_buffer, model.xi_buffer)
	model.queue(model.xi_norm_kernel, sum(model.R), nothing, model.K, model.xi_buffer)
end

function train!(model::gpuCTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for GPU accelerated collaborative topic Poisson factorization variational Bayes algorithm."

	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .> 0)										|| throw(ArgumentError("Iteration parameters must be positive integers."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf)	|| throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))

	update_buffer!(model)

	for k in 1:iter
		for _ in 1:viter
			update_xi!(model)
			update_phi!(model)
			update_zayin!(model)
			update_gimel!(model)
			if sum([norm(model.gimel[d] - model.gimel_old[d]) for d in 1:model.M]) < model.M * vtol
				break
			end
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

	update_host!(model, 1)
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

