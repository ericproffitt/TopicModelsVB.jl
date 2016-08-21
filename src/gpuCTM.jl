type gpuCTM <: GPUTopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	B::Int
	corp::Corpus
	batches::Vector{UnitRange{Int}}
	topics::VectorList{Int}
	mu::Vector{Float32}
	sigma::Matrix{Float32}
	beta::Matrix{Float32}
	lambda::VectorList{Float32}
	vsq::VectorList{Float32}
	lzeta::Vector{Float32}
	phi::MatrixList{Float32}
	invsigma::Matrix{Float32}
	newbeta::Void
	Npsums::VectorList{Int}
	Jpsums::VectorList{Int}
	terms::VectorList{Int}
	counts::VectorList{Int}
	words::VectorList{Int}
	device::OpenCL.Device
	context::OpenCL.Context
	queue::OpenCL.CmdQueue
	mukern::OpenCL.Kernel
	betakern::OpenCL.Kernel
	betanormkern::OpenCL.Kernel
	newbetakern::OpenCL.Kernel
	lambdakern::OpenCL.Kernel
	vsqkern::OpenCL.Kernel
	lzetakern::OpenCL.Kernel
	phikern::OpenCL.Kernel
	phinormkern::OpenCL.Kernel
	Cbuf::OpenCL.Buffer{Int}
	Npsumsbuf::OpenCL.Buffer{Int}
	Jpsumsbuf::OpenCL.Buffer{Int}
	termsbuf::OpenCL.Buffer{Int}
	countsbuf::OpenCL.Buffer{Int}
	wordsbuf::OpenCL.Buffer{Int}
	newtontempbuf::OpenCL.Buffer{Float32}
	newtongradbuf::OpenCL.Buffer{Float32}
	newtoninvhessbuf::OpenCL.Buffer{Float32}
	mubuf::OpenCL.Buffer{Float32}
	sigmabuf::OpenCL.Buffer{Float32}
	invsigmabuf::OpenCL.Buffer{Float32}
	betabuf::OpenCL.Buffer{Float32}
	newbetabuf::OpenCL.Buffer{Float32}
	lambdabuf::OpenCL.Buffer{Float32}
	vsqbuf::OpenCL.Buffer{Float32}
	lzetabuf::OpenCL.Buffer{Float32}
	phibuf::OpenCL.Buffer{Float32}
	elbo::Float32
	newelbo::Float32

	function gpuCTM(corp::Corpus, K::Integer, batchsize::Integer=length(corp))
		@assert !isempty(corp)		
		@assert all(ispositive([K, batchsize]))
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]	
		
		batches = partition(1:M, batchsize)
		B = length(batches)

		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		lambda = [zeros(K) for _ in 1:M]
		vsq = [ones(K) for _ in 1:M]
		lzeta = zeros(M)
		phi = [ones(K, N[d]) / K for d in batches[1]]

		model = new(K, M, V, N, C, B, copy(corp), batches, topics, mu, sigma, beta, lambda, vsq, lzeta, phi)
		fixmodel!(model, check=false)
		
		for (b, batch) in enumerate(batches)
			model.phi = [ones(K, N[d]) / K for d in batch]
			updateNewELBO!(model, b)
		end
		model.phi = [ones(K, N[d]) / K for d in batches[1]]
		updateELBO!(model)
		return model
	end
end

function Elogpeta(model::gpuCTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::gpuCTM, d::Int, m::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[m]' * model.lambda[d], counts) + model.C[d] * model.lzeta[d]
	return x
end

function Elogpw(model::gpuCTM, d::Int, m::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[m] .* log(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::gpuCTM, d::Int)
	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::gpuCTM, d::Int, m::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[m][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::gpuCTM)
	model.elbo = model.newelbo
	model.newelbo = 0
	return model.elbo
end

function updateNewELBO!(model::gpuCTM, b::Int)
	batch = model.batches[b]
	for (m, d) in enumerate(batch)
		model.newelbo += (Elogpeta(model, d)
						+ Elogpz(model, d, m)
						+ Elogpw(model, d, m)
						- Elogqeta(model, d)
						- Elogqz(model, d, m))					 
	end		
end

const CTM_MU_cpp =
"""
kernel void
updateMu(long K,
			long M,
			const global float *lambda,
			global float *mu)
		
			{
			long i = get_global_id(0);

			float acc = 0.0f;

			for (long d=0; d<M; d++)
				acc += lambda[K * d + i];

			mu[i] = acc / M;
			}
			"""

function updateMu!(model::gpuCTM)
	OpenCL.call(model.queue, model.mukern, model.K, nothing, model.K, model.M, model.lambdabuf, model.mubuf)
end

function updateSigma!(model::gpuCTM)
	@host model.mubuf
	@host model.lambdabuf
	@host model.vsqbuf

	model.sigma = diagm(sum(model.vsq)) / model.M + cov(hcat(model.lambda...)', mean=model.mu', corrected=false)
	(log(cond(model.sigma)) < 14) || (model.sigma += eye(model.K) * (eigmax(model.sigma) - 14 * eigmin(model.sigma)) / 13)
	model.invsigma = inv(model.sigma)

	@buf model.sigma
	@buf model.invsigma
end

const CTM_BETA_cpp =
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

const CTM_BETA_NORM_cpp =
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

function updateBeta!(model::gpuCTM)
	OpenCL.call(model.queue, model.betakern, (model.K, model.V), nothing, model.K, model.newbetabuf, model.betabuf)
	OpenCL.call(model.queue, model.betanormkern, model.K, nothing, model.K, model.V, model.betabuf)
end

const CTM_NEWBETA_cpp =
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

function updateNewbeta!(model::gpuCTM)
	OpenCL.call(model.queue, model.newbetakern, (model.K, model.V), nothing, model.K, model.Jpsumsbuf, model.countsbuf, model.wordsbuf, model.phibuf, model.newbetabuf)
end

const CTM_LAMBDA_cpp =
"""
$(RREF_cpp)
$(NORM2_cpp)

kernel void
updateLambda(long niter,
				float ntol,
				long K,
				long F,
				global float *A,
				global float *lambdaGrad,
				global float *lambdaInvHess,
				const global long *C,
				const global long *Npsums,
				const global long *counts,
				const global float *mu,
				const global float *sigma,
				const global float *invsigma,
				const global float *vsq,
				const global float *lzeta,
				const global float *phi,
				global float *lambda)
	
				{
				long d = get_global_id(0);

				long D = K * K * d;

				for (long _=0; _<niter; _++)
				{
					for (long i=0; i<K; i++)
					{
						float acc = 0.0f;

						for (long l=0; l<K; l++)
						{
							acc += invsigma[K * l + i] * (mu[l] - lambda[K * (F + d) + l]);
							A[D + K * l + i] = -C[d] * sigma[K * l + i] * exp(lambda[K * (F + d) + l] + 0.5f * vsq[K * (F + d) + l] - lzeta[F + d]);
						}

						for (long n=Npsums[d]; n<Npsums[d+1]; n++)
							acc += phi[K * n + i] * counts[n];

						lambdaGrad[K * d + i] = acc - C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - lzeta[F + d]);
						A[D + K * i + i] -= 1.0f;
					}

					for (long l=0; l<K; l++)
						for (long i=0; i<K; i++)
							lambdaInvHess[D + K * l + i] = sigma[K * l + i];

					rref(K, D, A, lambdaInvHess);		

					for (long l=0; l<K; l++)
						for (long i=0; i<K; i++)
							lambda[K * (F + d) + i] -= lambdaInvHess[D + K * l + i] * lambdaGrad[K * d + l];

					float lgnorm = norm2(K, d, lambdaGrad);
					if (lgnorm < ntol)
						break;
				}
				}
				"""

function updateLambda!(model::gpuCTM, b::Int, niter::Int, ntol::Float32)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.lambdakern, length(batch), nothing, niter, ntol, model.K, batch[1] - 1, model.newtontempbuf, model.newtongradbuf, model.newtoninvhessbuf, model.Cbuf, model.Npsumsbuf, model.countsbuf, model.mubuf, model.sigmabuf, model.invsigmabuf, model.vsqbuf, model.lzetabuf, model.phibuf, model.lambdabuf)
end

const CTM_VSQ_cpp =
"""
$(NORM2_cpp)

kernel void
updateVsq(long niter,
			float ntol,
			long K,
			long F,
			global float *p,
			global float *vsqGrad,
			const global long *C,
			const global float *invsigma,
			const global float *lambda,
			const global float *lzeta,
			global float *vsq)
			
			{
			long d = get_global_id(0);

			float vsqInvHess;

			for (long _=0; _<niter; _++)
			{
				float rho = 1.0f;

				for (long i=0; i<K; i++)
				{
					vsqGrad[K * d + i] = -0.5f * (invsigma[K * i + i] + C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - lzeta[F + d]) - 1 / vsq[K * (F + d) + i]);
					vsqInvHess = -1 / (0.25f * C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - lzeta[F + d]) + 0.5f / (vsq[K * (F + d) + i] * vsq[K * (F + d) + i]));
				
					p[K * d + i] = vsqGrad[K * d + i] * vsqInvHess;
					while (vsq[K * (F + d) + i] - rho * p[K * d + i] <= 0)
						rho *= 0.5f;
				}

				for (long i=0; i<K; i++)
					vsq[K * (F + d) + i] -= rho * p[K * d + i];

				float vgnorm = norm2(K, d, vsqGrad);
				if (vgnorm < ntol)
					break;
			}

			for (long i=0; i<K; i++)
				vsq[K * (F + d) + i] += $(EPSILON32);
			}
			"""

function updateVsq!(model::gpuCTM, b::Int, niter::Int, ntol::Float32)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.vsqkern, length(batch), nothing, niter, ntol, model.K, batch[1] - 1, model.newtontempbuf, model.newtongradbuf, model.Cbuf, model.invsigmabuf, model.lambdabuf, model.lzetabuf, model.vsqbuf)
end

const CTM_LZETA_cpp = 
"""
kernel void
updateLzeta(long K,
			long F,
			const global float *lambda,
			const global float *vsq,
			global float *lzeta)

			{
			long d = get_global_id(0);

			float maxval = 0.0f;

			for (long i=0; i<K; i++)
			{
				float x = lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i];
				if (x > maxval)
					maxval = x;
			}

			float acc = 0.0f;

			for (long i=0; i<K; i++)
				acc += exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - maxval);

			lzeta[F + d] = maxval + log(acc);
			}
			"""

function updateLzeta!(model::gpuCTM, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.lzetakern, length(batch), nothing, model.K, batch[1] - 1, model.lambdabuf, model.vsqbuf, model.lzetabuf)
end

const CTM_PHI_cpp =
"""
kernel void
updatePhi(long K,
			long F,
			const global long *Npsums,
			const global long *terms,
			const global float *beta,
			const global float *lambda,
			global float *phi)
	
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long n=Npsums[d]; n<Npsums[d+1]; n++)
				phi[K * n + i] = log(beta[K * terms[n] + i]) + lambda[K * (F + d) + i];
			}
			"""

const CTM_PHI_NORM_cpp =
"""
kernel void
normalizePhi(long K,
				global float *phi)
				
				{
				long dn = get_global_id(0);

				float maxval = 0.0f;

				for (long i=0; i<K; i++)				
					if (phi[K * dn + i] > maxval)
						maxval = phi[K * dn + i];

				float normalizer = 0.0f;
											
				for (long i=0; i<K; i++)
				{
					phi[K * dn + i] -= maxval;
					normalizer += exp(phi[K * dn + i]);
				}

				for (long i=0; i<K; i++)
					phi[K * dn + i] = exp(phi[K * dn + i]) / normalizer;
				}
				"""

function updatePhi!(model::gpuCTM, b::Int)
	batch = model.batches[b]
	OpenCL.call(model.queue, model.phikern, (model.K, length(batch)), nothing, model.K, batch[1] - 1, model.Npsumsbuf, model.termsbuf, model.betabuf, model.lambdabuf, model.phibuf)
	OpenCL.call(model.queue, model.phinormkern, sum(model.N[batch]), nothing, model.K, model.phibuf)
end

function train!(model::gpuCTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	niter, ntol = Int(niter), Float32(ntol)
	lowVRAM = model.B > 1	
	
	for k in 1:iter
		chk = (k % chkelbo == 0)
		for b in 1:model.B
			for _ in 1:viter
				oldlambda = @host model.lambdabuf
				updateLzeta!(model, b)
				updatePhi!(model, b)
				updateLambda!(model, b, niter, ntol)
				updateVsq!(model, b, niter, ntol)
				lambda = @host model.lambdabuf
				if sum([norm(diff) for diff in oldlambda - lambda]) < length(model.batches[b]) * vtol
					break
				end
			end
			updateNewbeta!(model)
			if chk
				updateHost!(model, b)
				updateNewELBO!(model, b)
			end
			lowVRAM && updateBuf!(model, b)
		end
		updateMu!(model)
		updateSigma!(model)
		updateBeta!(model)
		if checkELBO!(model, k, chk, tol)
			break
		end
	end
	updateHost!(model, 1)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

