type gpuCTM <: GPUTopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float64}
	sigma::Matrix{Float64}
	invsigma::Matrix{Float64}
	beta::Matrix{Float64}
	lambda::VectorList{Float64}
	vsq::VectorList{Float64}
	lzeta::Vector{Float64}
	phi::MatrixList{Float64}
	elbo::Float64

	function gpuCTM(corp::Corpus, K::Integer)
		@assert ispositive(K)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]	
		
		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = eye(K)
		invsigma = eye(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		lambda = [zeros(K) for _ in 1:M]
		vsq = [ones(K) for _ in 1:M]
		lzeta = zeros(M)
		phi = [ones(K, N[d]) / K for d in 1:M]

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, beta, lambda, vsq, lzeta, phi)
		updateELBO!(model)
		return model
	end
end

function Elogpeta(model::gpuCTM, d::Int)
	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::gpuCTM, d::Int)
	counts = model.corp[d].counts
	x = dot(model.phi[d]' * model.lambda[d], counts) + model.C[d] * model.lzeta[d]
	return x
end

function Elogpw(model::gpuCTM, d::Int)
	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::gpuCTM, d::Int)
	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::gpuCTM, d::Int)
	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function updateELBO!(model::gpuCTM)
	model.elbo = 0
	for d in 1:model.M
		model.elbo += (Elogpeta(model, d)
					+ Elogpz(model, d)
					+ Elogpw(model, d)
					- Elogqeta(model, d)
					- Elogqz(model, d))					 
	end		
	return model.elbo
end

const gpuCTMmucpp =
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
	model.mu = sum(model.lambda) / model.M
end

const gpuCTMsigmacpp =
"""
kernel void
updateSigma(long K,
			long M,
			const global float *mu,
			const global float *lambda,
			const global float *vsq,
			global float *sigma)

			{
			long i = get_global_id(0);
			lond l = get_global_id(1);

			float acc = 0.0f;

			for (long d=0; d<M; d++)
				acc += (lambda[K * d + i] - mu[i]) * (lambda[K * d + l] - mu[l]);

			if (i == l)
			{
				accvsq = 0.0f;

				for (d=0; d<M; d++)
					accvsq += vsq[K * d + i];

				acc += accvsq;
			}

			sigma[K * l + i] = acc / M;
			}
			"""

const gpuCTMinvsigmacpp =
"""
kernel void
updateInvsigma(long K,
				const global float *sigma,
				global float *invsigma)
				
				{
				long i = get_global_id(0);
				long l = get_global_id(1);

				}
				"""

function updateSigma!(model::gpuCTM)
	model.sigma = diagm(sum(model.vsq)) / model.M + cov(hcat(model.lambda...)', mean=model.mu', corrected=false)
	(log(cond(model.sigma)) < 14) || (model.sigma += eye(model.K) * (eigmax(model.sigma) - 14 * eigmin(model.sigma)) / 13)
	model.invsigma = inv(model.sigma)
end

const gpuCTMbetacpp =
"""
kernel void
updateBeta(long K,
			const global long *Jpsums,
			const global long *counts,
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

const gpuCTMbetanormcpp =
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
	model.beta = zeros(model.K, model.V)
	for d in 1:model.M
		terms, counts = model.corp[d].terms, model.corp[d].counts
		model.beta[:,terms] += model.phi[d] .* counts'
	end
	model.beta ./= sum(model.beta, 2)
end

const gpuCTMlambdacpp =
"""
kernel void
updateLambda(long niter,
				long ntol,
				long K, 
				const global long *C,
				const global float *mu,
				const global float *sigma,
				const global float *invsigma,
				const global float *vsq,
				const global float *lzeta,
				const global float *phi,
				global float *lambda)
	
				{
				long i = get_global_id(0);
				long d = get_global_id(1);

				for (long _=0; _<niter; _++)
				{
					local float A[K * K];
					A[K * i + i] = -1.0f;

					local float lambdaGrad[K];
					lambda[i] = -C[d] * exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - lzeta[d]);

					for (long l=0; l<K; l++)
					{
						lambdaGrad -= invsigma[K * l + i] * (lambda[K * d + l] - mu[l]);
						A[K * l + i] = -C * sigma[K * l + l] * exp(lambda[K * d + l] + 0.5f * vsq[K * d + l] - lzeta[l]);				
					}

					for (long n=Npsums[d]; n<Npsums[d+1]; n++)
						lambdaGrad += phi[K * n + i] * counts[n];

					barrier(CLK_LOCAL_MEM_FENCE);

					float lambdaInvHess = gaussElim(A, sigma);	

					for (long l=0; l<K; l++)
						lambda[K * l + i] -= lambdaInvHess[K * l + i] * lambdaGrad[l];

					barrier(CLK_LOCAL_MEM_FENCE);

					if (norm(lambdaGrad) < ntol)
						break;
				}
				}
				"""

function updateLambda!(model::gpuCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	counts = model.corp[d].counts
	for _ in 1:niter
		lambdaGrad = (-model.invsigma * (model.lambda[d] - model.mu) + model.phi[d] * counts - model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]))
		lambdaInvHess = -inv(eye(model.K) + model.C[d] * model.sigma * diagm(exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]))) * model.sigma
		model.lambda[d] -= lambdaInvHess * lambdaGrad
		if norm(lambdaGrad) < ntol
			break
		end
	end
end

const gpuCTMvsqcpp =
"""
kernel void
updateVsq(long niter,
			float ntol,
			long K,
			const global float *C,
			const global float *invsigma,
			const global float *lambda,
			const global float *lzeta,
			global float *vsq)
			
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long _=0; _<niter, _++)
			{
				float rho = 1.0f;

				local float vsqGrad[K];
				local float vsqInvHess[K];
				local float p[K];

				vsqGrad[i] = -0.5 * invsigma[K * i + i] + C[d] * exp(lambda[K * d + i] + 0.5 * vsq[K * d + i] - lzeta[d]) - 1 / vsq[K * d + i];
				vsqInvHess[i] = -1 / (0.25f * C[d] * exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - lzeta[d]) + 0.5f / (vsq[K * d + i] * vsq[K * d + i]));

				p = vsqGrad[i] * vsqInvHess[i];

				barrier(CLK_LOCAL_MEM_FENCE);

				for ()


			}

			}
			"""

function updateVsq!(model::gpuCTM, d::Int, niter::Integer, ntol::Real)
	"Newton's method."

	for _ in 1:niter
		rho = 1.0
		vsqGrad = -0.5 * (diag(model.invsigma) + model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]) - 1 ./ model.vsq[d])
		vsqInvHess = -1 ./ (0.25 * model.C[d] * exp(model.lambda[d] + 0.5 * model.vsq[d] - model.lzeta[d]) + 0.5 ./ model.vsq[d].^2)
		p = vsqInvHess .* vsqGrad
		
		while minimum(model.vsq[d] - rho * p) <= 0
			rho *= 0.5
		end	
		model.vsq[d] -= rho * p
		
		if norm(vsqGrad) < ntol
			break
		end
	end
	@bumper model.vsq[d]
end

const gpuCTMlzetacpp = 
"""
kernel void
updateLzeta(long K,
			const global float *lambda,
			const global float *vsq,
			global float *lzeta)

			{
			long d = get_global_id(0);

			float acc = 0.0f;

			for (long i=0; i<K; i++)
				acc += exp(lambda[K * d + i] + 0.5 * vsq[K * d + i]);

			lzeta[d] = log(acc);
			}
			"""

function updateLzeta!(model::gpuCTM, d::Int)
	model.lzeta[d] = logsumexp(model.lambda[d] + 0.5 * model.vsq[d])	
end

const gpuCTMphicpp =
"""
kernel void
updatePhi(long K,
			const global long *terms,
			const global float *beta,
			const global float *lambda,
			global float *phi)
	
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			}
			"""

function updatePhi!(model::gpuCTM, d::Int)
	terms = model.corp[d].terms
	model.phi[d] = addlogistic(log(model.beta[:,terms]) .+ model.lambda[d], 1)
end

function train!(model::gpuCTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	@assert all(!isnegative([tol, ntol, vtol]))
	@assert all(ispositive([iter, niter, viter, chkelbo]))
	fixmodel!(model)	
	
	for k in 1:iter
		for d in 1:model.M
			for _ in 1:viter
				oldlambda = model.lambda[d]				
				updateLambda!(model, d, niter, ntol)
				updateVsq!(model, d, niter, ntol)
				updateLzeta!(model, d)
				updatePhi!(model, d)
				if norm(oldlambda - model.lambda[d]) < vtol
					break
				end
			end
		end
		updateMu!(model)
		updateSigma!(model)
		updateBeta!(model)
		if checkELBO!(model, k, chkelbo, tol)
			break
		end
	end
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end

