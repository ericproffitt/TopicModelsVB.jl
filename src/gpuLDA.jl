mutable struct gpuLDA <: TopicModel
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
	gamma::VectorList{Float32}
	phi::MatrixList{Float32}
	newbeta::Void
	Npsums::VectorList{Int}
	Jpsums::VectorList{Int}
	terms::VectorList{Int}
	counts::VectorList{Int}
	words::VectorList{Int}
	device::cl.Device
	context::cl.Context
	queue::cl.CmdQueue
	betakern::cl.Kernel
	betanormkern::cl.Kernel
	newbetakern::cl.Kernel
	gammakern::cl.Kernel
	phikern::cl.Kernel
	phinormkern::cl.Kernel
	Elogthetakern::cl.Kernel
	Elogthetasumkern::cl.Kernel
	Npsumsbuf::cl.Buffer{Int}
	Jpsumsbuf::cl.Buffer{Int}
	termsbuf::cl.Buffer{Int}
	countsbuf::cl.Buffer{Int}
	wordsbuf::cl.Buffer{Int}
	alphabuf::cl.Buffer{Float32}
	betabuf::cl.Buffer{Float32}
	newbetabuf::cl.Buffer{Float32}
	gammabuf::cl.Buffer{Float32}
	phibuf::cl.Buffer{Float32}
	Elogthetabuf::cl.Buffer{Float32}
	Elogthetasumbuf::cl.Buffer{Float32}
	elbo::Float32

	function gpuLDA(corp::Corpus, K::Integer, batchsize::Integer=length(corp))
		K > 0 || throw(ArgumentError("Number of topics must be a positive integer."))

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = copy(beta)
		beta_temp = zeros(K, V)
		Elogtheta = [-Base.MathConstants.eulergamma * ones(K) .- digamma(K) for _ in 1:M]
		Elogtheta_old = copy(Elogtheta)
		gamma = [ones(K) for _ in 1:M]
		phi = [ones(K, N[d]) / K for d in 1:M]
		elbo = 0

		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, beta_old, beta_temp, Elogtheta, Elogtheta_old, gamma, phi, elbo)

		model.terms = vcat([doc.terms for doc in model.corp) .- 1
		model.counts = vcat([doc.counts for doc in model.corp)
		model.words = sortperm(model.terms) .- 1
			
		J = zeros(Int, model.V)
		for j in model.terms
			J[j+1] += 1
		end

		model.device, model.context, model.queue = cl.create_compute_context()

		beta_program = cl.Program(model.context, source=LDA_BETA_c) |> cl.build!
		beta_norm_program = cl.Program(model.context, source=LDA_BETA_NORM_c) |> cl.build!
		new_beta_program = cl.Program(model.context, source=LDA_NEWBETA_c) |> cl.build!
		gamma_program = cl.Program(model.context, source=LDA_GAMMA_c) |> cl.build!
		phi_program = cl.Program(model.context, source=LDA_PHI_c) |> cl.build!
		phi_norm_program = cl.Program(model.context, source=LDA_PHI_NORM_c) |> cl.build!
		Elogtheta_program = cl.Program(model.context, source=LDA_ELOGTHETA_c) |> cl.build!

		model.beta_kernel = cl.Kernel(beta_program, "update_beta")
		model.beta_norm_kernel = cl.Kernel(beta_norm_program, "normalizeBeta")
		model.new_beta_kernel = cl.Kernel(newbetaprog, "updateNewbeta")
		model.gamma_kernel = cl.Kernel(gammaprog, "updateGamma")
		model.phi_kernel = cl.Kernel(phiprog, "update_phi")
		model.phi_norm_kernel = cl.Kernel(phinormprog, "normalize_phi")
		model.Elogtheta_kernel = cl.Kernel(Elogthetaprog, "update_Elogtheta")

		@buffer model.alpha
		@buffer model.beta
		@buffer model.gamma
		@buffer model.Elogthetasum
		@buffer model.newbeta
		update_buffer!(model)

		model.phi = [ones(K, N[d]) / K for d in 1:M]
		model.Elogtheta = [digamma.(ones(K)) - digamma(K) for _ in 1:M]
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
	x = dot(model.phi * counts, model.Elogtheta)
	return x
end

function Elogpw(model::gpuLDA, d::Int)
	"Compute E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log.(@boink model.beta[:,terms]) * counts)
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
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::gpuLDA)
	"Update the evidence lower bound."

	for d in 1:model.M
		model.elbo += Elogptheta(model, d) + Elogpz(model, d) + Elogpw(model, d) - Elogqtheta(model, d) - Elogqz(model, d)
	end
end

function updateAlpha!(model::gpuLDA, niter::Integer, ntol::Real)
	"Update alpha."
	"Interior-point Newton's method with log-barrier and back-tracking line search."

	@host model.alpha_buffer
	@host model.Elogthetasum_buffer

	Elogtheta_sum = sum([model.Elogtheta[d] for d in 1:model.M])

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
			global float *newbeta,
			global float *beta)
	
			{
			long i = get_global_id(0);
			long j = get_global_id(1);

			beta[K * j + i] = newbeta[K * j + i];
			newbeta[K * j + i] = 0.0f;
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

function updateBeta!(model::gpuLDA)
	"Update beta"
	"Analytic."

	model.queue(model.betakern, (model.K, model.V), nothing, model.K, model.new_beta_buffer, model.beta_buffer)
	model.queue(model.betanormkern, model.K, nothing, model.K, model.V, model.beta_buffer)
end

const LDA_NEWBETA_c =
"""
kernel void
updateNewbeta(	long K,
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
	model.queue(model.newbetakern, (model.K, model.V), nothing, model.K, model.Jpsumsbuf, model.counts_buffer, model.words_buffer, model.phi_buffer, model.new_beta_buffer)
end

const LDA_ELOGTHETA_c =
"""
$(DIGAMMA_c)

kernel void
update_Elogtheta(	long F,
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

function updateElogtheta!(model::gpuLDA)
	"Update E[log(theta)]."
	"Analytic."
	
	model.queue(model.Elogtheta_kernel, model.M, nothing, 0, model.K, model.gamma_buffer, model.Elogtheta_buffer)
end

const LDA_GAMMA_c =
"""
kernel void
update_gamma(	long F,
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

function updateGamma!(model::gpuLDA)
	"Update gamma."
	"Analytic."

	model.queue(model.gammakern, (model.K, model.M), nothing, 0, model.K, model.Npsumsbuf, model.counts_buffer, model.alpha_buffer, model.phi_buffer, model.gamma_buffer)
end

const LDA_PHI_c =
"""
kernel void
update_phi(	long K,
			const global,
			long *Npsums,
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

	model.queue(model.phi_kernel, (model.K, model.M), nothing, model.K, model.terms_buffer, model.beta_buffer, model.Elogtheta_buffer, model.phi_buffer)	
	model.queue(model.phi_norm_kernel, sum(model.N), nothing, model.K, model.phi_buffer)
end

function train!(model::gpuLDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, check_elbo::Real=1)
	"Coordinate ascent optimization procedure for GPU accelerated latent Dirichlet allocation variational Bayes algorithm."

	all([tol, ntol, vtol] .>= 0) || throw(ArgumentError("Tolerance parameters must be nonnegative."))
	all([iter, niter, viter] .> 0) || throw(ArgumentError("Iteration parameters must be positive integers."))
	(isa(check_elbo, Integer) & (check_elbo > 0)) | (check_elbo == Inf)  || throw(ArgumentError("check_elbo parameter must be a positive integer or Inf."))

	for k in 1:iter
		for _ in 1:viter
			update_phi!(model)			
			update_gamma!(model)
			update_Elogtheta!(model)
			if sum([norm(diff) for diff in model.Elogtheta - model.Elogtheta_old]) < vtol
				break
			end
		end
		update_beta!(model)
		update_alpha!(model, niter, ntol)
		
		if check_elbo!(model, check_elbo, k, tol)
			break
		end
	end

	update_host!(model)
	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end











