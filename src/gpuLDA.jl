mutable struct gpuLDA <: TopicModel
	"GPU accelerated latent Dirichlet allocation."

	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float32}
	beta::Matrix{Float32}
	Elogtheta::VectorList{Float32}
	Elogtheta_old::VectorList{Float32}
	gamma::VectorList{Float32}
	phi::MatrixList{Float32}
	elbo::Float32
	device::cl.Device
	context::cl.Context
	queue::cl.CmdQueue
	beta_kernel::cl.Kernel
	beta_norm_kernel::cl.Kernel
	Elogtheta_kernel::cl.Kernel
	gamma_kernel::cl.Kernel
	phi_kernel::cl.Kernel
	phi_norm_kernel::cl.Kernel
	N_partial_sums_buffer::cl.Buffer{Int}
	J_partial_sums_buffer::cl.Buffer{Int}
	terms_buffer::cl.Buffer{Int}
	terms_sortperm_buffer::cl.Buffer{Int}
	counts_buffer::cl.Buffer{Int}
	alpha_buffer::cl.Buffer{Float32}
	beta_buffer::cl.Buffer{Float32}
	gamma_buffer::cl.Buffer{Float32}
	Elogtheta_buffer::cl.Buffer{Float32}
	Elogtheta_old_buffer::cl.Buffer{Float32}
	Elogtheta_dist_buffer::cl.Buffer{Float32}
	phi_buffer::cl.Buffer{Float32}

	function gpuLDA(corp::Corpus, K::Integer)
		check_corp(corp)
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = copy(beta)
		beta_temp = zeros(K, V)
		Elogtheta = [ones(K) for _ in 1:M]#[-Base.MathConstants.eulergamma * ones(K) .- digamma(K) for _ in 1:M]
		Elogtheta_old = copy(Elogtheta)
		gamma = [ones(K) for _ in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:M]
		elbo = 0

		device, context, queue = cl.create_compute_context()

		beta_program = cl.Program(context, source=LDA_BETA_c) |> cl.build!
		beta_norm_program = cl.Program(context, source=LDA_BETA_NORM_c) |> cl.build!
		Elogtheta_program = cl.Program(context, source=LDA_ELOGTHETA_c) |> cl.build!
		gamma_program = cl.Program(context, source=LDA_GAMMA_c) |> cl.build!
		phi_program = cl.Program(context, source=LDA_PHI_c) |> cl.build!
		phi_norm_program = cl.Program(context, source=LDA_PHI_NORM_c) |> cl.build!

		beta_kernel = cl.Kernel(beta_program, "update_beta")
		beta_norm_kernel = cl.Kernel(beta_norm_program, "normalize_beta")
		Elogtheta_kernel = cl.Kernel(Elogtheta_program, "update_Elogtheta")
		gamma_kernel = cl.Kernel(gamma_program, "update_gamma")
		phi_kernel = cl.Kernel(phi_program, "update_phi")
		phi_norm_kernel = cl.Kernel(phi_norm_program, "normalize_phi")

		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, Elogtheta, Elogtheta_old, gamma, phi, elbo, device, context, queue, beta_kernel, beta_norm_kernel, Elogtheta_kernel, gamma_kernel, phi_kernel, phi_norm_kernel)
		update_elbo!(model)	
		return model
	end
end

function Elogptheta(model::gpuLDA, d::Int)
	"Compute E[log(P(theta))]."

	x = loggamma(sum(model.alpha)) - sum(loggamma.(model.alpha)) + dot(model.alpha .- 1, model.Elogtheta[d])
	return x
end

function Elogpz(model::gpuLDA, d::Int)
	"Compute E[log(P(z))]."

	counts = model.corp[d].counts
	x = dot(model.phi[d] * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::gpuLDA, d::Int)
	"Compute E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi[d] .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqtheta(model::gpuLDA, d::Int)
	"Compute E[log(q(theta))]."

	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::gpuLDA, d::Int)
	"Compute E[log(q(z))]."

	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[d][:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::gpuLDA)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		model.elbo += Elogptheta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqtheta(model, d) - Elogqz(model, d)
	end

	return model.elbo
end

function update_alpha!(model::gpuLDA, niter::Integer, ntol::Real)
	"Update alpha."
	"Interior-point Newton's method with log-barrier and back-tracking line search."

	Elogtheta_host = cl.read(model.queue, model.Elogtheta_buffer)[1:(model.K * model.M)]

	Elogtheta_sum = zeros(Float32, model.K)
	for d in 1:model.M, i in 1:model.K
		Elogtheta_sum[i] += Elogtheta_host[model.K * (d - 1) + i]
	end

	#Elogtheta_host = reshape(cl.read(model.queue, model.Elogtheta_buffer), model.K, model.M + 64 - model.M % 64)[:,1:model.M]
	#Elogtheta_sum = vec(sum(Elogtheta_host, dims=2))
	#Elogtheta_sum = sum([model.Elogtheta[d] for d in 1:model.M])

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alpha_grad = [nu / model.alpha[i] + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] .+ Elogtheta_sum
		alpha_invhess_diag = -1 ./ (model.M * trigamma.(model.alpha) + nu ./ model.alpha.^2)
		p = (alpha_grad .- dot(alpha_grad, alpha_invhess_diag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(alpha_invhess_diag))) .* alpha_invhess_diag
		
		while minimum(model.alpha - rho * p) < 0
			rho *= 0.5
		end	
		model.alpha -= rho * p
		
		if (norm(alpha_grad) < ntol) & (nu / model.K < ntol)
			break
		end
		nu *= 0.5
	end
	@bumper model.alpha
	@buffer model.alpha
end

const LDA_BETA_c =
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

const LDA_BETA_NORM_c =
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

function update_beta!(model::gpuLDA)
	"Update beta"
	"Analytic."

	model.queue(model.beta_kernel, (model.K, model.V), nothing, model.K, model.J_partial_sums_buffer, model.terms_sortperm_buffer, model.counts_buffer, model.phi_buffer, model.beta_buffer)
	model.queue(model.beta_norm_kernel, model.K, nothing, model.K, model.V, model.beta_buffer)
end

const LDA_ELOGTHETA_c =
"""
$(DIGAMMA_c)

kernel void
update_Elogtheta(	long K,
					long M,
					const global float *gamma,
					global float *Elogtheta,
					global float *Elogtheta_dist)

					{
					long d = get_global_id(0);

					float acc1 = 0.0f;
					float acc2 = 0.0f;

					for (long i=0; i<K; i++)
						acc1 += gamma[K * d + i];

					acc1 = digamma(acc1);

					for (long i=0; i<K; i++)
					{
						float x = digamma(gamma[K * d + i]) - acc1;

						acc2 += (x - Elogtheta[K * d + i]) * (x - Elogtheta[K * d + i]);
						Elogtheta[K * d + i] = x;
					}

					Elogtheta_dist[d] = sqrt(acc2);
					}
					"""

function update_Elogtheta!(model::gpuLDA)
	"Update E[log(theta)]."
	"Analytic."
	
	#model.Elogtheta_old = model.Elogtheta

	model.queue(model.Elogtheta_kernel, model.M, nothing, model.K, model.M, model.gamma_buffer, model.Elogtheta_buffer, model.Elogtheta_dist_buffer)
	
	Elogtheta_dist_host = cl.read(model.queue, model.Elogtheta_dist_buffer)[1:model.M]
	return Elogtheta_dist_host

	#Elogtheta_host = cl.read(model.queue, model.Elogtheta_buffer)[1:(model.K * model.M)]
	#Elogtheta_host = reshape(Elogtheta_host, model.K, model.M)
	#model.Elogtheta = [Elogtheta_host[:,d] for d in 1:model.M]
	#return Elogtheta_host
	#@host model.Elogtheta_buffer
end

const LDA_GAMMA_c =
"""
kernel void
update_gamma(	long K,
				const global long *N_partial_sums,
				const global long *counts,
				const global float *alpha,
				const global float *phi,
				global float *gamma)

				{   
				long i = get_global_id(0);
				long d = get_global_id(1);

				float acc = 0.0f;

				for (long n=N_partial_sums[d]; n<N_partial_sums[d+1]; n++)
					acc += phi[K * n + i] * counts[n]; 

				gamma[K * d + i] = alpha[i] + acc + $(EPSILON32);
				}
				"""

function update_gamma!(model::gpuLDA)
	"Update gamma."
	"Analytic."

	model.queue(model.gamma_kernel, (model.K, model.M), nothing, model.K, model.N_partial_sums_buffer, model.counts_buffer, model.alpha_buffer, model.phi_buffer, model.gamma_buffer)
end

const LDA_PHI_c =
"""
kernel void
update_phi(	long K,
			const global long *N_partial_sums,
			const global long *terms,
			const global float *beta,
			const global float *Elogtheta,
			global float *phi)

			{
			long i = get_global_id(0);
			long d = get_global_id(1);

			for (long n=N_partial_sums[d]; n<N_partial_sums[d+1]; n++)
				phi[K * n + i] = beta[K * terms[n] + i] * exp(Elogtheta[K * d + i]);
			}
			"""

const LDA_PHI_NORM_c =
"""
kernel void
normalize_phi(	long K,
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

function update_phi!(model::gpuLDA)
	"Update phi."
	"Analytic."

	model.queue(model.phi_kernel, (model.K, model.M), nothing, model.K, model.N_partial_sums_buffer, model.terms_buffer, model.beta_buffer, model.Elogtheta_buffer, model.phi_buffer)	
	model.queue(model.phi_norm_kernel, sum(model.N), nothing, model.K, model.phi_buffer)
end

function train!(model::gpuLDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for GPU accelerated latent Dirichlet allocation variational Bayes algorithm."

	#check_model(model)
	all([tol, ntol, vtol] .>= 0)										|| throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .>= 0)										|| throw(ArgumentError("Iteration parameters must be nonnegative."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf) || throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))
	#all([isempty(doc) for doc in model.corp]) ? (iter = 0) : update_buffer!(model)
	#update_elbo!(model)

	#Elogtheta_host_old = hcat(model.Elogtheta...)

	for k in 1:iter
		for v in 1:viter
			update_phi!(model)			
			update_gamma!(model)
			#update_Elogtheta!(model)
			Elogtheta_dist_host = update_Elogtheta!(model)

			println(sum(Elogtheta_dist_host))

			if sum(Elogtheta_dist_host) < model.M * vtol
				break
			end

			#Elogtheta_dist = zeros(Float32, model.M)
			#for d in 1:model.M, i in 1:model.K
			#	Elogtheta_dist[d] += sqrt(sum((Elogtheta_host[model.K * (d - 1) + i] - Elogtheta_host_old[model.K * (d - 1) + i]).^2))
			#end

			#if sum(Elogtheta_dist) < model.M * vtol
			#if norm(Elogtheta_host - Elogtheta_host_old) < sqrt(model.M) * vtol
			#if median(sqrt.(sum((Elogtheta_host - Elogtheta_host_old).^2, dims=1))) < vtol
			#	Elogtheta_host_old = Elogtheta_host
			#if sum([norm(model.Elogtheta[d] - model.Elogtheta_old[d]) for d in 1:model.M]) < model.M * vtol
				#nothing
				#break
			#end
			#Elogtheta_host_old = Elogtheta_host
		end
		update_beta!(model)
		update_alpha!(model, niter, ntol)
		
		if check_elbo!(model, check_elbo, k, tol)
			break
		end
	end

	#(iter > 0) && update_host!(model)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end