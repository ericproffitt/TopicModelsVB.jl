type gpuCTPF <: GPUTopicModel
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
	newalef::Void
	newhe::Void
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
	device::OpenCL.Device
	context::OpenCL.Context
	queue::OpenCL.CmdQueue
	Npsumsbuf::OpenCL.Buffer{Int}
	Jpsumsbuf::OpenCL.Buffer{Int}
	Rpsumsbuf::OpenCL.Buffer{Int}
	Ypsumsbuf::OpenCL.Buffer{Int}
	termsbuf::OpenCL.Buffer{Int}
	countsbuf::OpenCL.Buffer{Int}
	wordsbuf::OpenCL.Buffer{Int}
	readersbuf::OpenCL.Buffer{Int}
	ratingsbuf::OpenCL.Buffer{Int}
	viewsbuf::OpenCL.Buffer{Int}
	alefkern::OpenCL.Kernel
	newalefkern::OpenCL.Kernel
	betkern::OpenCL.Kernel
	gimelkern::OpenCL.Kernel
	daletkern::OpenCL.Kernel
	hekern::OpenCL.Kernel
	newhekern::OpenCL.Kernel
	vavkern::OpenCL.Kernel
	zayinkern::OpenCL.Kernel
	hetkern::OpenCL.Kernel
	phikern::OpenCL.Kernel
	phinormkern::OpenCL.Kernel
	xikern::OpenCL.Kernel
	xinormkern::OpenCL.Kernel
	alefbuf::OpenCL.Buffer{Float32}
	newalefbuf::OpenCL.Buffer{Float32}
	betbuf::OpenCL.Buffer{Float32}
	gimelbuf::OpenCL.Buffer{Float32}
	daletbuf::OpenCL.Buffer{Float32}
	hebuf::OpenCL.Buffer{Float32}
	newhebuf::OpenCL.Buffer{Float32}
	vavbuf::OpenCL.Buffer{Float32}
	zayinbuf::OpenCL.Buffer{Float32}
	hetbuf::OpenCL.Buffer{Float32}
	phibuf::OpenCL.Buffer{Float32}
	xibuf::OpenCL.Buffer{Float32}
	elbo::Float32
	newelbo::Float32

	function gpuCTPF(corp::Corpus, K::Integer, batchsize::Integer=length(corp), basemodel::Union{Void, BaseTopicModel}=nothing)
		@assert !isempty(corp)		
		@assert all(ispositive([K, batchsize]))
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		R = [length(doc.readers) for doc in corp]

		batches = partition(1:M, batchsize)
		B = length(batches)

		@assert ispositive(U)
		@assert isequal(collect(1:U), sort(collect(keys(corp.users))))
		libs = [Int[] for _ in 1:U]
		for u in 1:U, d in 1:M
			u in corp[d].readers && push!(libs[u], d)
		end

		a, b, c, d, e, f, g, h = fill(0.1, 8)

		if isa(basemodel, Union{AbstractLDA, AbstractCTM})
			@assert isequal(size(basemodel.beta), (K, V))
			alef = exp(basemodel.beta)
			topics = basemodel.topics		
		elseif isa(basemodel, Union{AbstractfLDA, AbstractfCTM})
			@assert isequal(size(basemodel.fbeta), (K, V))
			alef = exp(basemodel.fbeta)
			topics = basemodel.topics
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
		phi = [ones(K, N[d]) / K for d in batches[1]]
		xi = [ones(2K, R[d]) / 2K for d in batches[1]]

		model = new(K, M, V, U, N, C, R, B, copy(corp), batches, topics, zeros(M, U), libs, Vector[], Vector[], a, b, c, d, e, f, g, h, alef, bet, gimel, dalet, he, vav, zayin, het, phi, xi)
		fixmodel!(model, check=false)

		for (b, batch) in enumerate(batches)
			model.phi = [ones(K, N[d]) / K for d in batch]
			model.xi = [ones(2K, R[d]) / 2K for d in batch]
			updateNewELBO!(model, b)
		end
		model.phi = [ones(K, N[d]) / K for d in batches[1]]
		model.xi = [ones(2K, R[d]) / 2K for d in batches[1]]
		updateELBO!(model)
		return model
	end
end

gpuCTPF(corp::Corpus, K::Int, basemodel::Union{Void, BaseTopicModel}) = gpuCTPF(corp, K, length(corp), basemodel)

function Elogpya(model::gpuCTPF, d::Int, m::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[m][i,u])
		x += (ra * model.xi[m][i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.gimel[d][i] / model.dalet[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::gpuCTPF, d::Int, m::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[m][model.K+i,u])
		x += (ra * model.xi[m][model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.zayin[d][i] / model.het[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::gpuCTPF, d::Int, m::Int)
	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[m][i,n])
		x += (c * model.phi[m][i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.alef[i,j]) - log(model.bet[i]))
				- (model.gimel[d][i] / model.dalet[i]) * (model.alef[i,j] / model.bet[i]) - sum([pdf(binom, z) * lgamma(z + 1) for z in 0:c]))
	end
	return x
end

function Elogpbeta(model::gpuCTPF)
	x = model.V * model.K * (model.a * log(model.b) - lgamma(model.a))
	for j in 1:model.V, i in 1:model.K
		x += (model.a - 1) * (digamma(model.alef[i,j]) - log(model.bet[i])) - model.b * model.alef[i,j] / model.bet[i]
	end
	return x
end

function Elogptheta(model::gpuCTPF, d::Int)
	x = model.K * (model.c * log(model.d) - lgamma(model.c))
	for i in 1:model.K
		x += (model.c - 1) * (digamma(model.gimel[d][i]) - log(model.dalet[i])) - model.d * model.gimel[d][i] / model.dalet[i]
	end
	return x
end

function Elogpeta(model::gpuCTPF)
	x = model.U * model.K * (model.e * log(model.f) - lgamma(model.e))
	for u in 1:model.U, i in 1:model.K
		x += (model.e - 1) * (digamma(model.he[i,u]) - log(model.vav[i])) - model.f * model.he[i,u] / model.vav[i]
	end
	return x
end

function Elogpepsilon(model::gpuCTPF, d::Int)
	x = model.K * (model.g * log(model.h) - lgamma(model.g))
	for i in 1:model.K
		x += (model.g - 1) * (digamma(model.zayin[d][i]) - log(model.het[i])) - model.h * model.zayin[d][i] / model.het[i]
	end
	return x
end

function Elogqy(model::gpuCTPF, d::Int, m::Int)
	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[m][:,u]))
	end
	return x
end

function Elogqz(model::gpuCTPF, d::Int, m::Int)
	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[m][:,n]))
	end
	return x
end

function Elogqbeta(model::gpuCTPF)
	x = 0
	for j in 1:model.V, i in 1:model.K
		x -= entropy(Gamma(model.alef[i,j], 1 / model.bet[i]))
	end
	return x
end

function Elogqtheta(model::gpuCTPF, d::Int)
	x = 0
	for i in 1:model.K
		x -= entropy(Gamma(model.gimel[d][i], 1 / model.dalet[i]))
	end
	return x
end

function Elogqeta(model::gpuCTPF)
	x = 0
	for u in 1:model.U, i in 1:model.K
		x -= entropy(Gamma(model.he[i,u], 1 / model.vav[i]))
	end
	return x
end	

function Elogqepsilon(model::gpuCTPF, d::Int)
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

function updateNewELBO!(model::gpuCTPF, b::Int)
	batch = model.batches[b]
	for (m, d) in enumerate(batch)
		model.newelbo += (Elogpya(model, d, m)
						+ Elogpyb(model, d, m)
						+ Elogpz(model, d, m)
						+ Elogptheta(model, d)
						+ Elogpepsilon(model, d)
						- Elogqy(model, d, m)
						- Elogqz(model, d, m) 
						- Elogqtheta(model, d)
						- Elogqepsilon(model, d))
	end
end


const CTPF_ALEF_cpp =
"""
kernel void
updateAlef(long K,
			float a,
			global float *newalef,
			global float *alef)

			{
			long i = get_global_id(0);
			long j = get_global_id(1);

			alef[K * j + i] = newalef[K * j + i];
			newalef[K * j + i] = a;
			}
			"""

function updateAlef!(model::gpuCTPF)
	OpenCL.call(model.queue, model.alefkern, (model.K, model.V), nothing, model.K, model.a, model.newalefbuf, model.alefbuf)
end

const CTPF_NEWALEF_cpp =
"""
kernel void
updateNewalef(long K,
				const global long *Jpsums,
				const global long *counts,
				const global long *words,
				const global float *phi,
				global float *newalef)
							
				{
				long i = get_global_id(0);
				long j = get_global_id(1);	

				float acc = 0.0f;

				for (long w=Jpsums[j]; w<Jpsums[j+1]; w++)
					acc += counts[words[w]] * phi[K * words[w] + i];

				newalef[K * j + i] += acc;
				}
				"""

function updateNewalef!(model::gpuCTPF)
	OpenCL.call(model.queue, model.newalefkern, (model.K, model.V), nothing, model.K, model.Jpsumsbuf, model.countsbuf, model.wordsbuf, model.phibuf, model.newalefbuf)
end

const CTPF_BET_cpp = 
"""
kernel void
updateBet(long K,
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

function updateBet!(model::gpuCTPF)
	OpenCL.call(model.queue, model.betkern, model.K, nothing, model.K, model.M, model.b, model.alefbuf, model.gimelbuf, model.daletbuf, model.betbuf)
end

const CTPF_GIMEL_cpp = 
"""
kernel void
updateGimel(long F,
			long K,
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

			float accphi = 0.0f;
			float accxi = 0.0f;

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				accphi += phi[K * n + i] * counts[n];

			for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
				accxi += xi[2 * K * r + i] * ratings[r]; 

			gimel[K * (F + d) + i] = c + accphi + accxi;
			}
			"""

function updateGimel!(model::gpuCTPF, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.gimelkern, (model.K, length(batch)), nothing, batch[1] - 1, model.K, model.c, model.Npsumsbuf, model.Rpsumsbuf, model.countsbuf, model.ratingsbuf, model.phibuf, model.xibuf, model.gimelbuf)
end

const CTPF_DALET_cpp =
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

function updateDalet!(model::gpuCTPF)
	OpenCL.call(model.queue, model.daletkern, model.K, nothing, model.K, model.V, model.U, model.d, model.alefbuf, model.betbuf, model.hebuf, model.vavbuf, model.daletbuf)
end

const CTPF_HE_cpp =
"""
kernel void
updateHe(long K,
			float e,
			global float *newhe,
			global float *he)

			{
			long i = get_global_id(0);
			long u = get_global_id(1);

			he[K * u + i] = newhe[K * u + i];
			newhe[K * u + i] = e;
			}
			"""

function updateHe!(model::gpuCTPF)
	OpenCL.call(model.queue, model.hekern, (model.K, model.U), nothing, model.K, model.e, model.newhebuf, model.hebuf)
end

const CTPF_NEWHE_cpp =
"""
kernel void
updateNewhe(long K,
			const global long *Ypsums,
			const global long *ratings,
			const global long *views,
			const global float *xi,
			global float *newhe)

			{
			long i = get_global_id(0);
			long u = get_global_id(1);

			float acc = 0.0f;

			for (long r=Ypsums[u]; r<Ypsums[u+1]; r++)
				acc += ratings[views[r]] * (xi[2 * K * views[r] + i] + xi[K * (2 * views[r] + 1) + i]);

			newhe[K * u + i] += acc;
			}
			"""

function updateNewhe!(model::gpuCTPF)
	OpenCL.call(model.queue, model.newhekern, (model.K, model.U), nothing, model.K, model.Ypsumsbuf, model.ratingsbuf, model.viewsbuf, model.xibuf, model.newhebuf)
end

const CTPF_VAV_cpp = 
"""
kernel void
updateVav(long K,
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
	OpenCL.call(model.queue, model.vavkern, model.K, nothing, model.K, model.M, model.f, model.gimelbuf, model.daletbuf, model.zayinbuf, model.hetbuf, model.vavbuf)
end

const CTPF_ZAYIN_cpp =
"""
kernel void
updateZayin(long F,
			long K,
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

function updateZayin!(model::gpuCTPF, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.zayinkern, (model.K, length(batch)), nothing, batch[1] - 1, model.K, model.g, model.Rpsumsbuf, model.ratingsbuf, model.xibuf, model.zayinbuf)
end

const CTPF_HET_cpp =
"""
kernel void
updateHet(long K,
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

function updateHet!(model::gpuCTPF)
	OpenCL.call(model.queue, model.hetkern, model.K, nothing, model.K, model.U, model.h, model.hebuf, model.vavbuf, model.hetbuf)
end

const CTPF_PHI_cpp =
"""
$(DIGAMMA_cpp)

kernel void
updatePhi(long F,
			long K,
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

const CTPF_PHI_NORM_cpp =
"""
kernel void
normalizePhi(long K,
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
	OpenCL.call(model.queue, model.phikern, (model.K, length(batch)), nothing, batch[1] - 1, model.K, model.Npsumsbuf, model.termsbuf, model.alefbuf, model.betbuf, model.gimelbuf, model.daletbuf, model.phibuf)
	OpenCL.call(model.queue, model.phinormkern, sum(model.N[batch]), nothing, model.K, model.phibuf)
end

const CTPF_XI_cpp =
"""
$(DIGAMMA_cpp)

kernel void
updateXi(long F,
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

const CTPF_XI_NORM_cpp =
"""
kernel void
normalizeXi(long K,
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
	OpenCL.call(model.queue, model.xikern, (model.K, length(batch)), nothing, batch[1] - 1, model.K, model.Rpsumsbuf, model.readersbuf, model.betbuf, model.gimelbuf, model.daletbuf, model.hebuf, model.vavbuf, model.zayinbuf, model.hetbuf, model.xibuf)
	OpenCL.call(model.queue, model.xinormkern, sum(model.R[batch]), nothing, model.K, model.xibuf)
end

function train!(model::gpuCTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, vtol]))
	@assert all(ispositive([iter, viter, chkelbo]))
	lowVRAM = model.B > 1

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for b in 1:model.B
			for _ in 1:viter
				oldgimel = @host b model.gimelbuf
				updatePhi!(model, b)
				updateXi!(model, b)
				updateGimel!(model, b)
				updateZayin!(model, b)
				gimel = @host b model.gimelbuf
				if sum([norm(diff) for diff in oldgimel - gimel]) < length(model.batches[b]) * vtol
					break
				end
			end
			updateNewalef!(model)
			updateNewhe!(model)
			if chk
				updateHost!(model, b)
				updateNewELBO!(model, b)
			end
			lowVRAM && updateBuf!(model, b)
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
	updateHost!(model, 1)
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

