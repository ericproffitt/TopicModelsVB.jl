function check_model(model::gpuCTM)
	@assert isequal(vcat(model.batches...), collect(1:model.M))
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])	
	@assert all(isfinite.(model.mu))	
	@assert isequal(size(model.sigma), (model.K, model.K))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(length(model.lambda), model.M)
	@assert all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])	
	@assert isequal(length(model.vsq), model.M)
	@assert all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])
	@assert all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive.(model.vsq[d])) for d in 1:model.M])	
	@assert all(isfinite.(model.logzeta))	
	@assert isequal(length(model.phi), length(model.batches[1]))
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in model.batches[1]])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in model.batches[1]])

	model.invsigma = inv(model.sigma)
	model.newbeta = nothing

	model.terms = [vcat([doc.terms for doc in model.corp[batch]]...) - 1 for batch in model.batches]
	model.counts = [vcat([doc.counts for doc in model.corp[batch]]...) for batch in model.batches]
	model.words = [sortperm(termvec) - 1 for termvec in model.terms]

	model.Npsums = [zeros(Int, length(batch) + 1) for batch in model.batches]
	for (b, batch) in enumerate(model.batches)
		for (n, d) in enumerate(batch)
			model.Npsums[b][n+1] = model.Npsums[b][n] + model.N[d]
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

	model.device, model.context, model.queue = cl.create_compute_context()

	muprog = cl.Program(model.context, source=CTM_MU_cpp) |> cl.build!
	betaprog = cl.Program(model.context, source=CTM_BETA_cpp) |> cl.build!
	betanormprog = cl.Program(model.context, source=CTM_BETA_NORM_cpp) |> cl.build!
	newbetaprog = cl.Program(model.context, source=CTM_NEWBETA_cpp) |> cl.build!
	lambdaprog = cl.Program(model.context, source=CTM_LAMBDA_cpp) |> cl.build!
	vsqprog = cl.Program(model.context, source=CTM_VSQ_cpp) |> cl.build!
	logzetaprog = cl.Program(model.context, source=CTM_logzeta_cpp) |> cl.build!
	phiprog = cl.Program(model.context, source=CTM_PHI_cpp) |> cl.build!
	phinormprog = cl.Program(model.context, source=CTM_PHI_NORM_cpp) |> cl.build!

	model.mukern = cl.Kernel(muprog, "updateMu")
	model.betakern = cl.Kernel(betaprog, "updateBeta")
	model.betanormkern = cl.Kernel(betanormprog, "normalizeBeta")
	model.newbetakern = cl.Kernel(newbetaprog, "updateNewbeta")
	model.lambdakern = cl.Kernel(lambdaprog, "updateLambda")
	model.vsqkern = cl.Kernel(vsqprog, "updateVsq")
	model.logzetakern = cl.Kernel(logzetaprog, "updatelogzeta")
	model.phikern = cl.Kernel(phiprog, "updatePhi")
	model.phinormkern = cl.Kernel(phinormprog, "normalizePhi")
		
	@buf model.mu
	@buf model.sigma
	@buf model.beta
	@buf model.lambda
	@buf model.vsq
	@buf model.logzeta
	@buf model.invsigma
	@buf model.newbeta
	updateBuf!(model, 0)

	model.newelbo = 0
	nothing
end

mutable struct gpuCTM <: GPUTopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float32}
	sigma::Matrix{Float32}
	beta::Matrix{Float32}
	lambda::VectorList{Float32}
	vsq::VectorList{Float32}
	logzeta::Vector{Float32}
	phi::MatrixList{Float32}
	invsigma::Matrix{Float32}
	elbo::Float32
	device::cl.Device
	context::cl.Context
	queue::cl.CmdQueue
	mukern::cl.Kernel
	beta_kernel::cl.Kernel
	beta_norm_kernel::cl.Kernel
	lambda_kernel::cl.Kernel
	vsq_kernel::cl.Kernel
	logzeta_kernel::cl.Kernel
	phi_kernel::cl.Kernel
	phi_norm_kernel::cl.Kernel
	N_partial_sums_buffer::cl.Buffer{Int}
	J_partial_sums_buffer::cl.Buffer{Int}
	terms_buffer::cl.Buffer{Int}
	terms_sortperm_buffer::cl.Buffer{Int}
	counts_buffer::cl.Buffer{Int}
	mu_buffer::cl.Buffer{Float32}
	sigma_buffer::cl.Buffer{Float32}
	invsigma_buffer::cl.Buffer{Float32}
	beta_buffer::cl.Buffer{Float32}
	lambda_buffer::cl.Buffer{Float32}
	vsq_buffer::cl.Buffer{Float32}
	logzeta_buffer::cl.Buffer{Float32}
	phi_buffer::cl.Buffer{Float32}
	newtontempbuf::cl.Buffer{Float32}
	newtongradbuf::cl.Buffer{Float32}
	newtoninvhessbuf::cl.Buffer{Float32}

	function gpuCTM(corp::Corpus, K::Integer, batchsize::Integer=length(corp))
		@assert !isempty(corp)		
		@assert all(ispositive.([K, batchsize]))
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]	

		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = Matrix(I, K, K)
		invsigma = Matrix(I, K, K)
		beta = rand(Dirichlet(V, 1.0), K)'
		lambda = [zeros(K) for _ in 1:M]
		lambda_old = copy(lambda)
		vsq = [ones(K) for _ in 1:M]
		logzeta = fill(0.5, M)
		phi = [ones(K, N[d]) / K for d in 1:M]

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, beta, lambda, lambda_old, vsq, logzeta, phi, elbo)
		update_elbo!(model)
		return model
	end
end

function Elogpeta(model::gpuCTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::gpuCTM, d::Int, m::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[m]' * model.lambda[d], counts) + model.C[d] * model.logzeta[d]
	return x
end

function Elogpw(model::gpuCTM, d::Int, m::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[m] .* log.(@boink model.beta[:,terms]) * counts)
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
	model.queue(model.mukern, model.K, nothing, model.K, model.M, model.lambdabuf, model.mubuf)
end

function updateSigma!(model::gpuCTM)
	@host model.mubuf
	@host model.lambdabuf
	@host model.vsqbuf

	model.sigma = diagm(sum(model.vsq)) / model.M + Base.covm(hcat(model.lambda...), model.mu, 2, false)
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
	model.queue(model.betakern, (model.K, model.V), nothing, model.K, model.newbetabuf, model.betabuf)
	model.queue(model.betanormkern, model.K, nothing, model.K, model.V, model.betabuf)
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
	model.queue(model.newbetakern, (model.K, model.V), nothing, model.K, model.Jpsumsbuf, model.countsbuf, model.wordsbuf, model.phibuf, model.newbetabuf)
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
				const global float *logzeta,
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
							A[D + K * l + i] = -C[d] * sigma[K * l + i] * exp(lambda[K * (F + d) + l] + 0.5f * vsq[K * (F + d) + l] - logzeta[F + d]);
						}

						for (long n=Npsums[d]; n<Npsums[d+1]; n++)
							acc += phi[K * n + i] * counts[n];

						lambdaGrad[K * d + i] = acc - C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - logzeta[F + d]);
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
	model.queue(model.lambdakern, length(batch), nothing, niter, ntol, model.K, batch[1] - 1, model.newtontempbuf, model.newtongradbuf, model.newtoninvhessbuf, model.Cbuf, model.Npsumsbuf, model.countsbuf, model.mubuf, model.sigmabuf, model.invsigmabuf, model.vsqbuf, model.logzetabuf, model.phibuf, model.lambdabuf)
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
			const global float *logzeta,
			global float *vsq)
			
			{
			long d = get_global_id(0);

			float vsqInvHess;

			for (long _=0; _<niter; _++)
			{
				float rho = 1.0f;

				for (long i=0; i<K; i++)
				{
					vsqGrad[K * d + i] = -0.5f * (invsigma[K * i + i] + C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - logzeta[F + d]) - 1 / vsq[K * (F + d) + i]);
					vsqInvHess = -1 / (0.25f * C[d] * exp(lambda[K * (F + d) + i] + 0.5f * vsq[K * (F + d) + i] - logzeta[F + d]) + 0.5f / (vsq[K * (F + d) + i] * vsq[K * (F + d) + i]));
				
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
	model.queue(model.vsqkern, length(batch), nothing, niter, ntol, model.K, batch[1] - 1, model.newtontempbuf, model.newtongradbuf, model.Cbuf, model.invsigmabuf, model.lambdabuf, model.logzetabuf, model.vsqbuf)
end

const CTM_logzeta_cpp = 
"""
kernel void
updatelogzeta(long K,
			long F,
			const global float *lambda,
			const global float *vsq,
			global float *logzeta)

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

			logzeta[F + d] = maxval + log(acc);
			}
			"""

function updatelogzeta!(model::gpuCTM, b::Int)
	batch = model.batches[b]
	model.queue(model.logzetakern, length(batch), nothing, model.K, batch[1] - 1, model.lambdabuf, model.vsqbuf, model.logzetabuf)
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
	model.queue(model.phikern, (model.K, length(batch)), nothing, model.K, batch[1] - 1, model.Npsumsbuf, model.termsbuf, model.betabuf, model.lambdabuf, model.phibuf)
	model.queue(model.phinormkern, sum(model.N[batch]), nothing, model.K, model.phibuf)
end

function train!(model::gpuCTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(.!isnegative.([tol, ntol, vtol]))
	@assert all(ispositive.([iter, niter, viter, chkelbo]))
	niter, ntol = Int(niter), Float32(ntol)
	lowVRAM = model.B > 1	
	
	for k in 1:iter
		chk = (k % chkelbo == 0)
		for b in 1:model.B
			for _ in 1:viter
				oldlambda = @host model.lambdabuf
				updatelogzeta!(model, b)
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

