type gpuLDA <: GPUTopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	B::Int
	corp::Corpus
	batches::Vector{UnitRange{Int}}
	topics::VectorList{Int}
	alpha::Vector{Float32}
	beta::Matrix{Float32}
	gamma::VectorList{Float32}
	phi::MatrixList{Float32}
	Elogtheta::VectorList{Float32}
	Elogthetasum::Vector{Float32}
	newbeta::Void
	Npsums::VectorList{Int}
	Jpsums::VectorList{Int}
	terms::VectorList{Int}
	counts::VectorList{Int}
	words::VectorList{Int}
	device::OpenCL.Device
	context::OpenCL.Context
	queue::OpenCL.CmdQueue
	betakern::OpenCL.Kernel
	betanormkern::OpenCL.Kernel
	newbetakern::OpenCL.Kernel
	gammakern::OpenCL.Kernel
	phikern::OpenCL.Kernel
	phinormkern::OpenCL.Kernel
	Elogthetakern::OpenCL.Kernel
	Elogthetasumkern::OpenCL.Kernel
	Npsumsbuf::OpenCL.Buffer{Int}
	Jpsumsbuf::OpenCL.Buffer{Int}
	termsbuf::OpenCL.Buffer{Int}
	countsbuf::OpenCL.Buffer{Int}
	wordsbuf::OpenCL.Buffer{Int}
	alphabuf::OpenCL.Buffer{Float32}
	betabuf::OpenCL.Buffer{Float32}
	newbetabuf::OpenCL.Buffer{Float32}
	gammabuf::OpenCL.Buffer{Float32}
	phibuf::OpenCL.Buffer{Float32}
	Elogthetabuf::OpenCL.Buffer{Float32}
	Elogthetasumbuf::OpenCL.Buffer{Float32}
	elbo::Float32
	newelbo::Float32

	function gpuLDA(corp::Corpus, K::Integer, batchsize::Integer=length(corp))
		@assert !isempty(corp)		
		@assert all(ispositive([K, batchsize]))
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]

		batches = partition(1:M, batchsize)
		B = length(batches)

		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		newbeta = Array(Float32, 0, 0)
		gamma = [ones(K) for _ in 1:M]
		phi = [ones(K, N[d]) / K for d in batches[1]]		

		model = new(K, M, V, N, C, B, copy(corp), batches, topics, alpha, beta, gamma, phi)	
		fixmodel!(model, check=false)

		for (b, batch) in enumerate(batches)
			model.phi = [ones(K, N[d]) / K for d in batch]
			model.Elogtheta = [digamma(ones(K)) - digamma(K) for _ in batch]
			updateNewELBO!(model, b)
		end
		model.phi = [ones(K, N[d]) / K for d in batches[1]]
		model.Elogtheta = [digamma(ones(K)) - digamma(K) for _ in batches[1]]
		updateELBO!(model)	
		return model
	end
end

function Elogptheta(model::gpuLDA, d::Int, m::Int)
	x = lgamma(sum(model.alpha)) - sum(lgamma(model.alpha)) + dot(model.alpha - 1, model.Elogtheta[m])
	return x
end

function Elogpz(model::gpuLDA, d::Int, m::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[m] * counts, model.Elogtheta[m])
	return x
end

function Elogpw(model::gpuLDA, d::Int, m::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[m] .* log(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqtheta(model::gpuLDA, d::Int)
	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::gpuLDA, d::Int, m::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[m][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::gpuLDA)
	model.elbo = model.newelbo
	model.newelbo = 0
	return model.elbo
end

function updateNewELBO!(model::gpuLDA, b::Int)
	batch = model.batches[b]
	for (m, d) in enumerate(batch)
		model.newelbo += (Elogptheta(model, d, m)
						+ Elogpz(model, d, m)
						+ Elogpw(model, d, m) 
						- Elogqtheta(model, d)
						- Elogqz(model, d, m))
	end
end

function updateAlpha!(model::gpuLDA, niter::Integer, ntol::Real)
	"Interior-point Newton method with log-barrier and back-tracking line search."

	@host model.alphabuf
	@host model.Elogthetasumbuf

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alphaGrad = Float32[nu / model.alpha[i] + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] + model.Elogthetasum
		alphaHessDiag = -(model.M * trigamma(model.alpha) + (nu ./ model.alpha.^2))
		p = (alphaGrad - sum(alphaGrad ./ alphaHessDiag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(1 ./ alphaHessDiag))) ./ alphaHessDiag

		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alphaGrad) < ntol) & ((nu / model.K) < ntol)
			break
		end
		nu *= 0.5
	end
	@bumper model.alpha

	model.Elogthetasum = zeros(model.K)
	@buf model.alpha
	@buf model.Elogthetasum
end

const LDA_BETA_cpp =
"""
kernel void
updateBeta(long K,
			global float *newbeta,
			global float *beta)
	
			{
			long i = get_global_id(0);
			long j = get_global_id(1);

			beta[K * j + i] = newbeta[K * j + i];
			newbeta[K * j + i] = 0.0f;
			}
			"""

const LDA_BETA_NORM_cpp =
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
	OpenCL.call(model.queue, model.betakern, (model.K, model.V), nothing, model.K, model.newbetabuf, model.betabuf)
	OpenCL.call(model.queue, model.betanormkern, model.K, nothing, model.K, model.V, model.betabuf)
end

const LDA_NEWBETA_cpp =
"""
kernel void
updateNewbeta(long K,
				const global long *Jpsums,
				const global long *counts,
				const global long *words,
				const global float *phi,
				global float *newbeta)
							
				{
				long i = get_global_id(0);
				long j = get_global_id(1);

				float acc = 0.0f;

				for (long w=Jpsums[j]; w<Jpsums[j+1]; w++)
					acc += counts[words[w]] * phi[K * words[w] + i];

				newbeta[K * j + i] += acc;
				}
				"""

function updateNewBeta!(model::gpuLDA)
	OpenCL.call(model.queue, model.newbetakern, (model.K, model.V), nothing, model.K, model.Jpsumsbuf, model.countsbuf, model.wordsbuf, model.phibuf, model.newbetabuf)
end

const LDA_GAMMA_cpp =
"""
kernel void
updateGamma(long F,
			long K,
			const global long *Npsums,
			const global long *counts,
			const global float *alpha,
			const global float *phi,
			global float *gamma)

			{   
			long i = get_global_id(0);
			long d = get_global_id(1);

			float acc = 0.0f;

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				acc += phi[K * n + i] * counts[n]; 

			gamma[K * (F + d) + i] = alpha[i] + acc + $(EPSILON32);
			}
			"""

function updateGamma!(model::gpuLDA, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.gammakern, (model.K, length(batch)), nothing, batch[1] - 1, model.K, model.Npsumsbuf, model.countsbuf, model.alphabuf, model.phibuf, model.gammabuf)
end

const LDA_PHI_cpp =
"""
kernel void
updatePhi(long K,
			const global long *Npsums,
			const global long *terms,
			const global float *beta,
			const global float *Elogtheta,
			global float *phi)
                                        
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				phi[K * n + i] = beta[K * terms[n] + i] * exp(Elogtheta[K * d + i]);
			}
			"""

const LDA_PHI_NORM_cpp =
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

function updatePhi!(model::gpuLDA, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.phikern, (model.K, length(batch)), nothing, model.K, model.Npsumsbuf, model.termsbuf, model.betabuf, model.Elogthetabuf, model.phibuf)	
	OpenCL.call(model.queue, model.phinormkern, sum(model.N[batch]), nothing, model.K, model.phibuf)
end

const LDA_ELOGTHETA_cpp =
"""
$(DIGAMMA_cpp)

kernel void
updateElogtheta(long F,
				long K,
				const global float *gamma,
				global float *Elogtheta)

				{
				long d = get_global_id(0);

				float acc = 0.0f;

				for (long i=0; i<K; i++)
					acc += gamma[K * (F + d) + i];

				acc = digamma(acc);	

				for (long i=0; i<K; i++)
					Elogtheta[K * d + i] = digamma(gamma[K * (F + d) + i]) - acc;
				}
				"""

function updateElogtheta!(model::gpuLDA, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.Elogthetakern, length(batch), nothing, batch[1] - 1, model.K, model.gammabuf, model.Elogthetabuf)
end

const LDA_ELOGTHETASUM_cpp =
"""
kernel void
updateElogthetasum(long K,
					long D,
					const global float *Elogtheta,
					global float *Elogthetasum)

					{
					long i = get_global_id(0);

					float acc = 0.0f;

					for (long d=0; d<D; d++)
						acc += Elogtheta[K * d + i];

					Elogthetasum[i] += acc;	
					}
					"""

function updateElogthetasum!(model::gpuLDA, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.Elogthetasumkern, model.K, nothing, model.K, length(batch), model.Elogthetabuf, model.Elogthetasumbuf)
end

function train!(model::gpuLDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Int=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	lowVRAM = model.B > 1

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for b in 1:model.B
			for _ in 1:viter
				oldElogtheta = @host b model.Elogthetabuf
				updateElogtheta!(model, b)
				updatePhi!(model, b)			
				updateGamma!(model, b)
				Elogtheta = @host b model.Elogthetabuf
				if sum([norm(diff) for diff in oldElogtheta - Elogtheta]) < length(model.batches[b]) * vtol
					break
				end
			end
			updateElogtheta!(model, b)
			updateElogthetasum!(model, b)
			updateNewBeta!(model)
			if chk
				updateHost!(model, b)
				updateNewELBO!(model, b)
			end
			lowVRAM && updateBuf!(model, b)
		end
		updateAlpha!(model, niter, ntol)
		updateBeta!(model)
		if checkELBO!(model, k, chk, tol)
			break
		end
	end
	updateHost!(model, 1)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end













