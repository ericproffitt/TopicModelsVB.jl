type gpuCTPF <: TopicModel
	K::Int
	M::Int
	V::Int
	U::Int
	N::Vector{Int}
	C::Vector{Int}
	R::Vector{Int}
	corp::Corpus
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
	device::OpenCL.Device
	context::OpenCL.Context
	queue::OpenCL.CmdQueue
	Npsums::OpenCL.Buffer{Int}
	Jpsums::OpenCL.Buffer{Int}
	Rpsums::OpenCL.Buffer{Int}
	Ypsums::OpenCL.Buffer{Int}
	terms::OpenCL.Buffer{Int}
	counts::OpenCL.Buffer{Int}
	words::OpenCL.Buffer{Int}
	readers::OpenCL.Buffer{Int}
	ratings::OpenCL.Buffer{Int}
	views::OpenCL.Buffer{Int}
	alefkern::OpenCL.Kernel
	betkern::OpenCL.Kernel
	gimelkern::OpenCL.Kernel
	daletkern::OpenCL.Kernel
	hekern::OpenCL.Kernel
	vavkern::OpenCL.Kernel
	zayinkern::OpenCL.Kernel
	hetkern::OpenCL.Kernel
	phikern::OpenCL.Kernel
	phinormkern::OpenCL.Kernel
	xikern::OpenCL.Kernel
	xinormkern::OpenCL.Kernel
	alefbuf::OpenCL.Buffer{Float32}
	betbuf::OpenCL.Buffer{Float32}
	gimelbuf::OpenCL.Buffer{Float32}
	daletbuf::OpenCL.Buffer{Float32}
	hebuf::OpenCL.Buffer{Float32}
	vavbuf::OpenCL.Buffer{Float32}
	zayinbuf::OpenCL.Buffer{Float32}
	hetbuf::OpenCL.Buffer{Float32}
	phibuf::OpenCL.Buffer{Float32}
	xibuf::OpenCL.Buffer{Float32}
	elbo::Float32

	function gpuCTPF(corp::Corpus, K::Integer, pmodel::Union{Void, BaseTopicModel}=nothing)
		@assert ispositive(K)		
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		R = [length(doc.readers) for doc in corp]

		@assert isequal(collect(1:U), sort(collect(keys(corp.users))))
		libs = [Int[] for _ in 1:U]
		for u in 1:U, d in 1:M
			u in corp[d].readers && push!(libs[u], d)
		end

		a, b, c, d, e, f, g, h = fill(0.1, 8)

		if isa(pmodel, Union{LDA, CTM, memLDA, memCTM, gpuLDA})
			@assert isequal(size(pmodel.beta), (K, V))
			alef = exp(pmodel.beta - 0.5)
			topics = pmodel.topics		
		elseif isa(pmodel, Union{fLDA, fCTM, memfLDA, memfCTM, gpuLDA})
			@assert isequal(size(pmodel.fbeta), (K, V))
			alef = exp(pmodel.fbeta - 0.5)
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
		phi = [rand(Dirichlet(K, 1.0), N[d]) for d in 1:M]
		xi = [rand(Dirichlet(2K, 1.0), R[d]) for d in 1:M]

		device, context, queue = OpenCL.create_compute_context()		

		terms = vcat([doc.terms for doc in corp]...) - 1
		counts = vcat([doc.counts for doc in corp]...)
		words = sortperm(terms) - 1

		readers = vcat([doc.readers for doc in corp]...) - 1
		ratings = vcat([doc.ratings for doc in corp]...)
		views = sortperm(readers) - 1

		Npsums = zeros(Int, M + 1)
		Rpsums = zeros(Int, M + 1)
		for d in 1:M
			Npsums[d+1] = Npsums[d] + N[d]
			Rpsums[d+1] = Rpsums[d] + R[d]
		end

		J = zeros(Int, V)
		for j in terms
			J[j+1] += 1
		end

		Jpsums = zeros(Int, V + 1)
		for j in 1:V
			Jpsums[j+1] = Jpsums[j] + J[j]
		end

		Y = zeros(Int, U)
		for r in readers
			Y[r+1] += 1
		end

		Ypsums = zeros(Int, U + 1)
		for u in 1:U
			Ypsums[u+1] = Ypsums[u] + Y[u]
		end

		Npsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=Npsums)
		Jpsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=Jpsums)
		terms = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=terms)
		counts = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=counts)
		words = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=words)

		Rpsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=Rpsums)
		Ypsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=Ypsums)
		readers = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=readers)
		ratings = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=ratings)
		views = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=views)

		alefprog = OpenCL.Program(context, source=CTPFalefcpp) |> OpenCL.build!
		betprog = OpenCL.Program(context, source=CTPFbetcpp) |> OpenCL.build!
		gimelprog = OpenCL.Program(context, source=CTPFgimelcpp) |> OpenCL.build!
		daletprog = OpenCL.Program(context, source=CTPFdaletcpp) |> OpenCL.build!
		heprog = OpenCL.Program(context, source=CTPFhecpp) |> OpenCL.build!
		vavprog = OpenCL.Program(context, source=CTPFvavcpp) |> OpenCL.build!
		zayinprog = OpenCL.Program(context, source=CTPFzayincpp) |> OpenCL.build!
		hetprog = OpenCL.Program(context, source=CTPFhetcpp) |> OpenCL.build!
		phiprog = OpenCL.Program(context, source=CTPFphicpp) |> OpenCL.build!
		phinormprog = OpenCL.Program(context, source=CTPFphinormcpp) |> OpenCL.build!
		xiprog = OpenCL.Program(context, source=CTPFxicpp) |> OpenCL.build!
		xinormprog = OpenCL.Program(context, source=CTPFxinormcpp) |> OpenCL.build!

		alefkern = OpenCL.Kernel(alefprog, "updateAlef")
		betkern = OpenCL.Kernel(betprog, "updateBet")
		gimelkern = OpenCL.Kernel(gimelprog, "updateGimel")
		daletkern = OpenCL.Kernel(daletprog, "updateDalet")
		hekern = OpenCL.Kernel(heprog, "updateHe")
		vavkern = OpenCL.Kernel(vavprog, "updateVav")
		zayinkern = OpenCL.Kernel(zayinprog, "updateZayin")
		hetkern = OpenCL.Kernel(hetprog, "updateHet")
		phikern = OpenCL.Kernel(phiprog, "updatePhi")
		phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
		xikern = OpenCL.Kernel(xiprog, "updateXi")
		xinormkern = OpenCL.Kernel(xinormprog, "normalizeXi")

		model = new(K, M, V, U, N, C, R, copy(corp), topics, zeros(M, U), libs, Vector[], Vector[], a, b, c, d, e, f, g, h, alef, bet, gimel, dalet, he, vav, zayin, het, phi, xi, device, context, queue, Npsums, Jpsums, Rpsums, Ypsums, terms, counts, words, readers, ratings, views, alefkern, betkern, gimelkern, daletkern, hekern, vavkern, zayinkern, hetkern, phikern, phinormkern, xikern, xinormkern)
		updateBuf!(model)
		updateELBO!(model)
		return model
	end
end

function Elogpya(model::gpuCTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[d][i,u])
		x += (ra * model.xi[d][i,u] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.gimel[d][i] / model.dalet[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpyb(model::gpuCTPF, d::Int)
	x = 0
	readers, ratings = model.corp[d].readers, model.corp[d].ratings
	for (u, (re, ra)) in enumerate(zip(readers, ratings)), i in 1:model.K
		binom = Binomial(ra, model.xi[d][model.K+i,u])
		x += (ra * model.xi[d][model.K+i,u] * (digamma(model.zayin[d][i]) - log(model.het[i]) + digamma(model.he[i,re]) - log(model.vav[i]))
				- (model.zayin[d][i] / model.het[i]) * (model.he[i,re] / model.vav[i]) - sum([pdf(binom, y) * lgamma(y + 1) for y in 0:ra]))
	end
	return x
end

function Elogpz(model::gpuCTPF, d::Int)
	x = 0
	terms, counts = model.corp[d].terms, model.corp[d].counts
	for (n, (j, c)) in enumerate(zip(terms, counts)), i in 1:model.K
		binom = Binomial(c, model.phi[d][i,n])
		x += (c * model.phi[d][i,n] * (digamma(model.gimel[d][i]) - log(model.dalet[i]) + digamma(model.alef[i,j]) - log(model.bet[i]))
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

function Elogqy(model::gpuCTPF, d::Int)
	x = 0
	for (u, ra) in enumerate(model.corp[d].ratings)
		x -= entropy(Multinomial(ra, model.xi[d][:,u]))
	end
	return x
end

function Elogqz(model::gpuCTPF, d::Int)
	x = 0
	for (n, c) in enumerate(model.corp[d].counts)
		x -= entropy(Multinomial(c, model.phi[d][:,n]))
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

const CTPFalefcpp =
"""
kernel void
updateAlef(long K,
			float a,
			const global long *Jpsums,
			const global long *counts,
			const global long *words,
			const global float *phi,
			global float *alef)
							
			{
			long i = get_global_id(0);
			long j = get_global_id(1);	

			float acc = 0.0f;

			for (long w=Jpsums[j]; w<Jpsums[j+1]; w++)
				acc += counts[words[w]] * phi[K * words[w] + i];

			alef[K * j + i] = a + acc;
			}
			"""

function updateAlef!(model::gpuCTPF)
	OpenCL.call(model.queue, model.alefkern, (model.K, model.V), nothing, model.K, model.a, model.Jpsums, model.counts, model.words, model.phibuf, model.alefbuf)
end

const CTPFbetcpp = 
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

const CTPFgimelcpp = 
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

			float accphi = 0.0f;
			float accxi = 0.0f;

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				accphi += phi[K * n + i] * counts[n];

			for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
				accxi += xi[2 * K * r + i] * ratings[r]; 

			gimel[K * d + i] = c + accphi + accxi;
			}
			"""

function updateGimel!(model::gpuCTPF)	
	OpenCL.call(model.queue, model.gimelkern, (model.K, model.M), nothing, model.K, model.c, model.Npsums, model.Rpsums, model.counts, model.ratings, model.phibuf, model.xibuf, model.gimelbuf)
end

const CTPFdaletcpp =
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

const CTPFhecpp =
"""
kernel void
updateHe(long K,
			float e,
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

			he[K * u + i] = e + acc;
			}
			"""

function updateHe!(model::gpuCTPF)
	OpenCL.call(model.queue, model.hekern, (model.K, model.U), nothing, model.K, model.e, model.Ypsums, model.ratings, model.views, model.xibuf, model.hebuf)
end

const CTPFvavcpp = 
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

const CTPFzayincpp =
"""
kernel void
updateZayin(long K,
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

			zayin[K * d + i] = g + acc; 
			}
			"""

function updateZayin!(model::gpuCTPF)
	OpenCL.call(model.queue, model.zayinkern, (model.K, model.M), nothing, model.K, model.g, model.Rpsums, model.ratings, model.xibuf, model.zayinbuf)
end

const CTPFhetcpp =
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

const CTPFphicpp =
"""
$(digammacpp)

kernel void
updatePhi(long K,
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

			float gdb = digamma(gimel[K * d + i]) - log(dalet[i]) - log(bet[i]);

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				phi[K * n + i] = exp(gdb + digamma(alef[K * terms[n] + i]));
			}
			"""

const CTPFphinormcpp =
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

function updatePhi!(model::gpuCTPF)
	OpenCL.call(model.queue, model.phikern, (model.K, model.M), nothing, model.K, model.Npsums, model.terms, model.alefbuf, model.betbuf, model.gimelbuf, model.daletbuf, model.phibuf)
	OpenCL.call(model.queue, model.phinormkern, sum(model.N), nothing, model.K, model.phibuf)
end

const CTPFxicpp =
"""
$(digammacpp)

kernel void
updateXi(long K,
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

			float gdv = digamma(gimel[K * d + i]) - log(dalet[i]) - log(bet[i]);
			float zhv = digamma(zayin[K * d + i]) - log(het[i]) - log(vav[i]);

			for (long r=Rpsums[d]; r<Rpsums[d+1]; r++)
			{
				xi[2 * K * r + i] = exp(gdv + digamma(he[K * readers[r] + i]));
				xi[K * (2 * r + 1) + i] = exp(zhv + digamma(he[K * readers[r] + i]));			
			}
			}
			"""

const CTPFxinormcpp =
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

function updateXi!(model::gpuCTPF)
	OpenCL.call(model.queue, model.xikern, (model.K, model.M), nothing, model.K, model.Rpsums, model.readers, model.betbuf, model.gimelbuf, model.daletbuf, model.hebuf, model.vavbuf, model.zayinbuf, model.hetbuf, model.xibuf)
	OpenCL.call(model.queue, model.xinormkern, sum(model.R), nothing, model.K, model.xibuf)
end

function train!(model::gpuCTPF; iter::Int=150, tol::Real=1.0, viter::Int=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, vtol]))
	@assert all(ispositive([iter, viter, chkelbo]))
	fixmodel!(model)

	for k in 1:iter
		for _ in 1:viter
			oldgimel = OpenCL.read(model.queue, model.gimelbuf)
			updatePhi!(model)
			updateXi!(model)
			updateGimel!(model)
			updateZayin!(model)
			if norm(oldgimel - OpenCL.read(model.queue, model.gimelbuf)) < model.M * vtol
				break
			end
		end
		updateDalet!(model)
		updateHet!(model)
		updateAlef!(model)
		updateBet!(model)
		updateHe!(model)
		updateVav!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
		println(k)
	end
	updateHost!(model)
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

