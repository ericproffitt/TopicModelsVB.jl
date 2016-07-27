type gpuLDA <: TopicModel
	K::Int
	M::Int
	V::Int
	C::Vector{Int}
	N::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float32}
	beta::Matrix{Float32}
	gamma::VectorList{Float32}
	phi::MatrixList{Float32}
	Elogtheta::VectorList{Float32}
	SUMElogtheta::Vector{Float32}
	device::OpenCL.Device
	context::OpenCL.Context
	queue::OpenCL.CmdQueue
	terms::OpenCL.Buffer{Int}
	counts::OpenCL.Buffer{Float32}
	Npsums::OpenCL.Buffer{Int}
	Jpsums::OpenCL.Buffer{Int}
	words::OpenCL.Buffer{Int}
	betakern::OpenCL.Kernel
	betanormkern::OpenCL.Kernel
	gammakern::OpenCL.Kernel
	phikern::OpenCL.Kernel
	phinormkern::OpenCL.Kernel
	Elogthetakern::OpenCL.Kernel
	SUMElogthetakern::OpenCL.Kernel
	alphabuf::OpenCL.Buffer{Float32}
	betabuf::OpenCL.Buffer{Float32}
	gammabuf::OpenCL.Buffer{Float32}
	phibuf::OpenCL.Buffer{Float32}
	Elogthetabuf::OpenCL.Buffer{Float32}
	SUMElogthetabuf::OpenCL.Buffer{Float32}
	elbo::Float32

	function gpuLDA(corp::Corpus, K::Integer)
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
		phi = [ones(K, N[d]) / K for d in 1:M]
		Elogtheta = [digamma(ones(K)) - digamma(K) for d in 1:M]
		SUMElogtheta = zeros(K)			

		device, context, queue = OpenCL.create_compute_context()		

		terms = vcat([doc.terms for doc in corp]...)
		words = collect(0:sum(N)-1)[sortperm(terms)]
		J = zeros(Int, V)
		for j in terms
			J[j] += 1
		end

		Npsums = Int[0]
		for d in 1:M
			push!(Npsums, Npsums[d] + N[d])
		end

		terms = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=terms - 1)	
		counts = OpenCL.Buffer(Float32, context, (:r, :copy), hostbuf=map(Float32, vcat([doc.counts for doc in corp]...)))		
		Npsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=Npsums)
		Jpsums = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=[sum(J[1:j]) for j in 0:V])
		words = OpenCL.Buffer(Int, context, (:r, :copy), hostbuf=words)		

		betaprog = OpenCL.Program(context, source=LDAbetacpp) |> OpenCL.build!
		betanormprog = OpenCL.Program(context, source=LDAbetanormcpp) |> OpenCL.build!
		gammaprog = OpenCL.Program(context, source=LDAgammacpp) |> OpenCL.build!
		phiprog = OpenCL.Program(context, source=LDAphicpp) |> OpenCL.build!
		phinormprog = OpenCL.Program(context, source=LDAphinormcpp) |> OpenCL.build!
		Elogthetaprog = OpenCL.Program(context, source=LDAElogthetacpp) |> OpenCL.build!
		SUMElogthetaprog = OpenCL.Program(context, source=LDASUMElogthetacpp) |> OpenCL.build!

		betakern = OpenCL.Kernel(betaprog, "updateBeta")
		betanormkern = OpenCL.Kernel(betanormprog, "normalizeBeta")
		gammakern = OpenCL.Kernel(gammaprog, "updateGamma")
		phikern = OpenCL.Kernel(phiprog, "updatePhi")
		phinormkern = OpenCL.Kernel(phinormprog, "normalizePhi")
		Elogthetakern = OpenCL.Kernel(Elogthetaprog, "updateElogtheta")
		SUMElogthetakern = OpenCL.Kernel(SUMElogthetaprog, "updateSUMElogtheta")

		model = new(K, M, V, C, N, copy(corp), topics, alpha, beta, gamma, phi, Elogtheta, SUMElogtheta, device, context, queue, terms, counts, Npsums, Jpsums, words, betakern, betanormkern, gammakern, phikern, phinormkern, Elogthetakern, SUMElogthetakern)
		updateBuf!(model)
		updateELBO!(model)
		return model
	end
end

function Elogptheta(model::gpuLDA, d::Int)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, model.Elogtheta[d])
	return x
end

function Elogpz(model::gpuLDA, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::gpuLDA, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log(model.beta[:,terms] + epsln) * counts)
	return x
end

function Elogqtheta(model::gpuLDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::gpuLDA, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::gpuLDA)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogptheta(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d) 
					- Elogqtheta(model, d)
					- Elogqz(model, d))
	end
	return model.elbo
end

function updateAlpha!(model::gpuLDA, niter::Integer, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	@host model.alphabuf
	@host model.SUMElogthetabuf

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphagrad = [(nu / model.alpha[i]) + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] + model.SUMElogtheta
		alphahessdiag = -(model.M * trigamma(model.alpha) + (nu ./ model.alpha.^2))
		p = (alphagrad - sum(alphagrad ./ alphahessdiag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(1 ./ alphahessdiag))) ./ alphahessdiag

		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alphagrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
	end
	@bumper model.alpha
	@buf model.alpha	
	
	model.SUMElogtheta = zeros(model.K)
	@buf model.SUMElogtheta
end

const LDAbetacpp =
"""
kernel void
updateBeta(long K,
			const global float *counts,
			const global long *Jpsums,
			const global long *words,
			const global float *phi,
			global float *beta)
							
			{
			long i = get_global_id(0);
			long j = get_global_id(1);	

			float acc = 0.0f;

			for (long w=Jpsums[j]; w<Jpsums[j+1]; w++)
				acc += counts[words[w]] * phi[K * words[w] + i];

			beta[K * j + i] = acc;
			}
			"""

const LDAbetanormcpp =
"""
kernel void
normalizeBeta(long K,
				long V,
				global float *beta)

				{   
				long i = get_global_id(0);

				float normalizer = 0.0f;
                                            
				for (long j=0; j<V; j++)
					normalizer += beta[K * j + i];

				for (long j=0; j<V; j++)
					beta[K * j + i] /= normalizer;
				}
				"""

function updateBeta!(model::gpuLDA)
	OpenCL.call(model.queue, model.betakern, (model.K, model.V), nothing, model.K, model.counts, model.Jpsums, model.words, model.phibuf, model.betabuf)
	OpenCL.call(model.queue, model.betanormkern, model.K, nothing, model.K, model.V, model.betabuf)
end

const LDAgammacpp =
"""
kernel void
updateGamma(long K,
			const global float *counts,
			const global long *Npsums,
			const global float *alpha,
			const global float *phi,
			global float *gamma)

			{   
			long i = get_global_id(0);
			long d = get_global_id(1);

			float acc = 0.0f;

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				acc += phi[K * n + i] * counts[n]; 

			gamma[K * d + i] = alpha[i] + acc;
			}
			"""

function updateGamma!(model::gpuLDA)
	OpenCL.call(model.queue, model.gammakern, (model.K, model.M), nothing, model.K, model.counts, model.Npsums, model.alphabuf, model.phibuf, model.gammabuf)
end

const LDAphicpp =
"""
kernel void
updatePhi(long K,
			const global long *terms,
			const global long *Npsums,
			const global float *beta,
			const global float *gamma,
			const global float *Elogtheta,
			global float *phi)
                                        
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				phi[K * n + i] = beta[K * terms[n] + i] * exp(Elogtheta[K * d + i]);
			}
			"""

const LDAphinormcpp =
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

function updatePhi!(model::gpuLDA)
	OpenCL.call(model.queue, model.phikern, (model.K, model.M), nothing, model.K, model.terms, model.Npsums, model.betabuf, model.gammabuf, model.Elogthetabuf, model.phibuf)	
	OpenCL.call(model.queue, model.phinormkern, sum(model.N), nothing, model.K, model.phibuf)
end

const LDAElogthetacpp =
"""
$(digammacpp)

kernel void
updateElogtheta(long K,
				const global float *gamma,
				global float *Elogtheta)

				{
				long d = get_global_id(0);

				float acc = 0.0f;

				for (long i=0; i<K; i++)
					acc += gamma[K * d + i];

				acc = digamma(acc);	

				for (long i=0; i<K; i++)
					Elogtheta[K * d + i] = digamma(gamma[K * d + i]) - acc;
				}
				"""

function updateElogtheta!(model::gpuLDA)
	OpenCL.call(model.queue, model.Elogthetakern, model.M, nothing, model.K, model.gammabuf, model.Elogthetabuf)
end

const LDASUMElogthetacpp =
"""
kernel void
updateSUMElogtheta(long K,
					long M,
					const global float *Elogtheta,
					global float *SUMElogtheta)

				{
				long i = get_global_id(0);

				float acc = 0.0f;

				for (long d=0; d<M; d++)
					acc += Elogtheta[K * d + i];

				SUMElogtheta[i] += acc;	
				}
				"""

function updateSUMElogtheta!(model::gpuLDA)
	OpenCL.call(model.queue, model.SUMElogthetakern, model.K, nothing, model.K, model.M, model.Elogthetabuf, model.SUMElogthetabuf)
end

function train!(model::gpuLDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)
	
	for k in 1:iter
		for _ in 1:viter
			oldgamma = OpenCL.read(model.queue, model.gammabuf)
			updateGamma!(model)
			updateElogtheta!(model)
			updatePhi!(model)
			if norm(oldgamma - OpenCL.read(model.queue, model.gammabuf)) < model.M * vtol
				break
			end
		end
		updateSUMElogtheta!(model)
		updateAlpha!(model, niter, ntol)
		updateBeta!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	updateHost!(model)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end
