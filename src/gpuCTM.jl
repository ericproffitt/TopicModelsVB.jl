mutable struct gpuCTM <: TopicModel
	"GPU accelerated correlated topic model."

	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	mu::Vector{Float32}
	sigma::Matrix{Float32}
	invsigma::Matrix{Float32}
	beta::Matrix{Float32}
	lambda::VectorList{Float32}
	lambda_dist::Vector{Float32}
	vsq::VectorList{Float32}
	logzeta::Vector{Float32}
	phi::MatrixList{Float32}
	elbo::Float32
	device::cl.Device
	context::cl.Context
	queue::cl.CmdQueue
	mu_kernel::cl.Kernel
	sigma_kernel::cl.Kernel
	beta_kernel::cl.Kernel
	beta_norm_kernel::cl.Kernel
	lambda_kernel::cl.Kernel
	vsq_kernel::cl.Kernel
	logzeta_kernel::cl.Kernel
	phi_kernel::cl.Kernel
	phi_norm_kernel::cl.Kernel
	C_buffer::cl.Buffer{Int}
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
	lambda_old_buffer::cl.Buffer{Float32}
	lambda_dist_buffer::cl.Buffer{Float32}
	vsq_buffer::cl.Buffer{Float32}
	logzeta_buffer::cl.Buffer{Float32}
	phi_buffer::cl.Buffer{Float32}
	newton_grad_buffer::cl.Buffer{Float32}
	newton_invhess_buffer::cl.Buffer{Float32}
	newton_temp_buffer::cl.Buffer{Float32}

	function gpuCTM(corp::Corpus, K::Integer)
		check_corp(corp)
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]	

		topics = [collect(1:V) for _ in 1:K]

		mu = zeros(K)
		sigma = Matrix(I, K, K)
		invsigma = Matrix(I, K, K)
		beta = rand(Dirichlet(V, 1.0), K)'
		lambda = [zeros(K) for _ in 1:M]
		lambda_dist = zeros(M)
		vsq = [ones(K) for _ in 1:M]
		logzeta = fill(0.5, M)
		phi = [ones(K, N[d]) / K for d in 1:M]
		elbo = 0

		device, context, queue = cl.create_compute_context()

		mu_program = cl.Program(context, source=CTM_MU_c) |> cl.build!
		sigma_program = cl.Program(context, source=CTM_SIGMA_c) |> cl.build!
		beta_program = cl.Program(context, source=CTM_BETA_c) |> cl.build!
		beta_norm_program = cl.Program(context, source=CTM_BETA_NORM_c) |> cl.build!
		lambda_program = cl.Program(context, source=CTM_LAMBDA_c) |> cl.build!
		vsq_program = cl.Program(context, source=CTM_VSQ_c) |> cl.build!
		logzeta_program = cl.Program(context, source=CTM_logzeta_c) |> cl.build!
		phi_program = cl.Program(context, source=CTM_PHI_c) |> cl.build!
		phi_norm_program = cl.Program(context, source=CTM_PHI_NORM_c) |> cl.build!

		mu_kernel = cl.Kernel(mu_program, "update_mu")
		sigma_kernel = cl.Kernel(sigma_program, "update_sigma")
		beta_kernel = cl.Kernel(beta_program, "update_beta")
		beta_norm_kernel = cl.Kernel(beta_norm_program, "normalize_beta")
		lambda_kernel = cl.Kernel(lambda_program, "update_lambda")
		vsq_kernel = cl.Kernel(vsq_program, "update_vsq")
		logzeta_kernel = cl.Kernel(logzeta_program, "update_logzeta")
		phi_kernel = cl.Kernel(phi_program, "update_phi")
		phi_norm_kernel = cl.Kernel(phi_norm_program, "normalize_phi")

		model = new(K, M, V, N, C, copy(corp), topics, mu, sigma, invsigma, beta, lambda, lambda_dist, vsq, logzeta, phi, elbo, device, context, queue, mu_kernel, sigma_kernel, beta_kernel, beta_norm_kernel, lambda_kernel, vsq_kernel, logzeta_kernel, phi_kernel, phi_norm_kernel)
		update_elbo!(model)
		return model
	end
end

function Elogpeta(model::gpuCTM, d::Int)
	"Compute E[log(P(eta))]."

	x = 0.5 * (logdet(model.invsigma) - model.K * log(2pi) - dot(diag(model.invsigma), model.vsq[d]) - dot(model.lambda[d] - model.mu, model.invsigma * (model.lambda[d] - model.mu)))
	return x
end

function Elogpz(model::gpuCTM, d::Int)
	"Compute E[log(P(z))]."

	counts = model.corp[d].counts
	x = dot(model.phi[d]' * model.lambda[d], counts) + model.C[d] * model.logzeta[d]
	return x
end

function Elogpw(model::gpuCTM, d::Int)
	"Compute E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqeta(model::gpuCTM, d::Int)
	"Compute E[log(q(eta))]."

	x = -entropy(MvNormal(model.lambda[d], diagm(model.vsq[d])))
	return x
end

function Elogqz(model::gpuCTM, d::Int)
	"Compute E[log(q(z))]."

	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::gpuCTM)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		model.elbo += Elogpeta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqeta(model, d) - Elogqz(model, d)			 
	end

	return model.elbo
end

const CTM_MU_c =
"""
kernel void
update_mu(	long K,
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

function update_mu!(model::gpuCTM)
	"Update mu."
	"Analytic."

	model.queue(model.mu_kernel, model.K, nothing, model.K, model.M, model.lambda_buffer, model.mu_buffer)
end

const CTM_SIGMA_c =
"""
kernel void
update_sigma( 	long K,
		        long M,
		        const global float *mu,
		        const global float *lambda,
		        const global float *vsq,
		        global float *sigma)
		      
		        {
		        long i = get_global_id(0);
		        long j = get_global_id(1);

		        float acc = 0.0f;

		        if (i == j)
		            for (long d=0; d<M; d++)
		              acc += vsq[K * d + i] + (lambda[K * d + i] - mu[i]) * (lambda[K * d + j] - mu[j]);

		        if (i != j)
		            for (long d=0; d<M; d++)
		              acc += (lambda[K * d + i] - mu[i]) * (lambda[K * d + j] - mu[j]);

		        sigma[K * i + j] = acc / M;
		        }
		        """

function update_sigma!(model::gpuCTM)
	"Update sigma."
	"Analytic"

	model.queue(model.sigma_kernel, (model.K, model.K), nothing, model.K, model.M, model.mu_buffer, model.lambda_buffer, model.vsq_buffer, model.sigma_buffer)

	@host model.sigma_buffer
	model.invsigma = inv(model.sigma)
	@buffer model.invsigma
end

const CTM_BETA_c =
"""
kernel void
update_beta(long K,
			const global long *J_partial_sums,
			const global long *terms_sortperm,
			const global long *counts,
			const global float *phi,
			global float *beta)
						
			{
			long i = get_global_id(0);
			long j = get_global_id(1);

			float acc = 0.0f;

			for (long w=J_partial_sums[j]; w<J_partial_sums[j+1]; w++)
				acc += counts[terms_sortperm[w]] * phi[K * terms_sortperm[w] + i];

			beta[K * j + i] = acc;
			}
			"""

const CTM_BETA_NORM_c =
"""
kernel void
normalize_beta(	long K,
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

function update_beta!(model::gpuCTM)
	"Update beta."
	"Analytic."

	model.queue(model.beta_kernel, (model.K, model.V), nothing, model.K, model.J_partial_sums_buffer, model.terms_sortperm_buffer, model.counts_buffer, model.phi_buffer, model.beta_buffer)
	model.queue(model.beta_norm_kernel, model.K, nothing, model.K, model.V, model.beta_buffer)
end

const CTM_LAMBDA_c =
"""
$(RREF_c)
$(NORM2_c)

kernel void
update_lambda(	long niter,
				float ntol,
				long K,
				global float *A,
				global float *lambda_old,
				global float *lambda_grad,
				global float *lambda_invhess,
				const global long *C,
				const global long *N_partial_sums,
				const global long *counts,
				const global float *mu,
				const global float *sigma,
				const global float *invsigma,
				const global float *vsq,
				const global float *logzeta,
				const global float *phi,
				global float *lambda,
				global float *lambda_dist)
	
				{
				long d = get_global_id(0);

				long D = K * K * d;

				for (long i=0; i<K; i++)
					lambda_old[K * d + i] = lambda[K * d + i];

				for (long _=0; _<niter; _++)
				{
					for (long i=0; i<K; i++)
					{
						float acc1 = 0.0f;

						for (long l=0; l<K; l++)
						{
							acc1 += invsigma[K * l + i] * (mu[l] - lambda[K * d + l]);
							A[D + K * l + i] = -C[d] * sigma[K * l + i] * exp(lambda[K * d + l] + 0.5f * vsq[K * d + l] - logzeta[d]);
						}

						for (long n=N_partial_sums[d]; n<N_partial_sums[d+1]; n++)
							acc1 += phi[K * n + i] * counts[n];

						lambda_grad[K * d + i] = acc1 - C[d] * exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - logzeta[d]);
						A[D + K * i + i] -= 1.0f;
					}

					for (long l=0; l<K; l++)
						for (long i=0; i<K; i++)
							lambda_invhess[D + K * l + i] = sigma[K * l + i];

					rref(K, D, A, lambda_invhess);		

					for (long l=0; l<K; l++)
						for (long i=0; i<K; i++)
							lambda[K * d + i] -= lambda_invhess[D + K * l + i] * lambda_grad[K * d + l];

					float lgnorm = norm2(K, d, lambda_grad);
					
					if (lgnorm < ntol)
						break;
				}

				float acc2 = 0.0f;

				for (long i=0; i<K; i++)
					acc2 += pow(lambda[K * d + i] - lambda_old[K * d + i], 2);

				lambda_dist[d] = sqrt(acc2);
				}
				"""

function update_lambda!(model::gpuCTM, niter::Int, ntol::Float32)
	"Update lambda."
	"Newton's method."

	model.queue(model.lambda_kernel, model.M, nothing, niter, ntol, model.K, model.newton_temp_buffer, model.lambda_old_buffer, model.newton_grad_buffer, model.newton_invhess_buffer, model.C_buffer, model.N_partial_sums_buffer, model.counts_buffer, model.mu_buffer, model.sigma_buffer, model.invsigma_buffer, model.vsq_buffer, model.logzeta_buffer, model.phi_buffer, model.lambda_buffer, model.lambda_dist_buffer)
	@host model.lambda_dist_buffer
end

const CTM_VSQ_c =
"""
$(NORM2_c)

kernel void
update_vsq(	long niter,
			float ntol,
			long K,
			global float *p,
			global float *vsq_grad,
			const global long *C,
			const global float *invsigma,
			const global float *lambda,
			const global float *logzeta,
			global float *vsq)
			
			{
			long d = get_global_id(0);

			float vsq_invhess;

			for (long _=0; _<niter; _++)
			{
				float rho = 1.0f;

				for (long i=0; i<K; i++)
				{
					vsq_grad[K * d + i] = -0.5f * (invsigma[K * i + i] + C[d] * exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - logzeta[d]) - 1 / vsq[K * d + i]);
					vsq_invhess = -1 / (0.25f * C[d] * exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - logzeta[d]) + 0.5f / (vsq[K * d + i] * vsq[K * d + i]));
				
					p[K * d + i] = vsq_grad[K * d + i] * vsq_invhess;
					while (vsq[K * d + i] - rho * p[K * d + i] <= 0)
						rho *= 0.5f;
				}

				for (long i=0; i<K; i++)
					vsq[K * d + i] -= rho * p[K * d + i];

				float vgnorm = norm2(K, d, vsq_grad);
				
				if (vgnorm < ntol)
					break;
			}

			for (long i=0; i<K; i++)
				vsq[K * d + i] += $(EPSILON32);
			}
			"""

function update_vsq!(model::gpuCTM, niter::Int, ntol::Float32)
	"Update vsq."
	"Interior-point Newton's method with log-barrier and back-tracking line search."

	model.queue(model.vsq_kernel, model.M, nothing, niter, ntol, model.K, model.newton_temp_buffer, model.newton_grad_buffer, model.C_buffer, model.invsigma_buffer, model.lambda_buffer, model.logzeta_buffer, model.vsq_buffer)
end

const CTM_logzeta_c = 
"""
kernel void
update_logzeta(	long K,
				const global float *lambda,
				const global float *vsq,
				global float *logzeta)

				{
				long d = get_global_id(0);

				float maxval = 0.0f;

				for (long i=0; i<K; i++)
				{
					float x = lambda[K * d + i] + 0.5f * vsq[K * d + i];
					if (x > maxval)
						maxval = x;
				}

				float acc = 0.0f;

				for (long i=0; i<K; i++)
					acc += exp(lambda[K * d + i] + 0.5f * vsq[K * d + i] - maxval);

				logzeta[d] = maxval + log(acc);
				}
				"""

function update_logzeta!(model::gpuCTM)
	"Update logzeta."
	"Analytic."

	model.queue(model.logzeta_kernel, model.M, nothing, model.K, model.lambda_buffer, model.vsq_buffer, model.logzeta_buffer)
end

const CTM_PHI_c =
"""
kernel void
update_phi(	long K,
			const global long *N_partial_sums,
			const global long *terms,
			const global float *beta,
			const global float *lambda,
			global float *phi)
	
			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long n=N_partial_sums[d]; n<N_partial_sums[d+1]; n++)
				phi[K * n + i] = log(beta[K * terms[n] + i]) + lambda[K * d + i];
			}
			"""

const CTM_PHI_NORM_c =
"""
kernel void
normalize_phi(	long K,
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

function update_phi!(model::gpuCTM)
	"Update phi."
	"Analytic."

	model.queue(model.phi_kernel, (model.K, model.M), nothing, model.K, model.N_partial_sums_buffer, model.terms_buffer, model.beta_buffer, model.lambda_buffer, model.phi_buffer)
	model.queue(model.phi_norm_kernel, sum(model.N), nothing, model.K, model.phi_buffer)
end

function train!(model::gpuCTM; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for GPU accelerated correlated topic model variational Bayes algorithm."

	ntol = Float32(ntol)
	check_model(model)
	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .>= 0)										|| throw(ArgumentError("Iteration parameters must be nonnegative."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf) || throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))
	all([isempty(doc) for doc in model.corp]) ? (iter = 0) : update_buffer!(model)
	update_elbo!(model)

	for k in 1:iter
		for v in 1:viter
			update_phi!(model)
			update_logzeta!(model)
			update_vsq!(model, niter, ntol)
			update_lambda!(model, niter, ntol)

			if median(model.lambda_dist) < vtol
				break
			end
		end
		update_beta!(model)
		update_sigma!(model)
		update_mu!(model)

		if check_elbo!(model, check_elbo, k, tol)
			break
		end
	end

	(iter > 0) && update_host!(model)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end