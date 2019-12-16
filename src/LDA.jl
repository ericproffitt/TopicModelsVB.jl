showdocs(model::TopicModel, doc_indices::Vector{Int}) = showdocs(model.corp, doc_indices)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, doc_range::UnitRange{Int}) = showdocs(model.corp, collect(doc_range))
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)

getlex(model::TopicModel) = sort(collect(values(model.corp.lex)))
getusers(model::TopicModel) = sort(collect(values(model.corp.users)))

Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")

function checkELBO!(model::TopicModel, k::Int, chk::Bool, tol::Real)
	converged = false
	if chk
		∆elbo = -(model.elbo - updateELBO!(model))
		println(k, " ∆elbo: ", round(∆elbo, 3))
		if abs(∆elbo) < tol
			converged = true
		end
	end

	return converged
end

function gendoc(model::AbstractLDA, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.beta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function showtopics{T<:Integer}(model::TopicModel, N::Integer=min(15, model.V); topics::Union{T, Vector{T}}=collect(1:model.K), cols::Integer=4)
	@assert checkbounds(Bool, 1:model.V, N)
	@assert checkbounds(Bool, 1:model.K, topics)
	@assert ispositive(cols)
	isa(topics, Vector) || (topics = [topics])
	cols = min(cols, length(topics))

	lex = model.corp.lex
	maxjspacings = [maximum([length(lex[j]) for j in topic[1:N]]) for topic in model.topics]

	for block in partition(topics, cols)
		for j in 0:N
			for (k, i) in enumerate(block)
				if j == 0
					jspacing = max(4, maxjspacings[i] - length("$i") - 2)
					k == cols ? yellow("topic $i") : yellow("topic $i" * " "^jspacing)
				else
					jspacing = max(6 + length("$i"), maxjspacings[i]) - length(lex[model.topics[i][j]]) + 4
					k == cols ? print(lex[model.topics[i][j]]) : print(lex[model.topics[i][j]] * " "^jspacing)
				end
			end
			println()
		end
		println()
	end
end

mutable struct LDA <: TopicModel
	K::Int
	M::Int
	V::Int
	N::Vector{Int}
	C::Vector{Int}
	corp::Corpus
	topics::VectorList{Int}
	alpha::Vector{Float64}
	beta::Matrix{Float64}
	beta_old::Matrix{Float64}
	beta_temp::Matrix{Float64}
	gamma::VectorList{Float64}
	Elogtheta::VectorList{Float64}
	Elogtheta_old::VectorList{Float64}
	phi::Matrix{Float64}

	elbo::Float64

	function LDA(corp::Corpus, K::Integer)
		@assert ispositive(K)
		@assert !isempty(corp)
		checkcorp(corp)

		M, V, U = size(corp)
		N = [length(doc) for doc in corp]
		C = [size(doc) for doc in corp]
		
		topics = [collect(1:V) for _ in 1:K]

		alpha = ones(K)
		beta = rand(Dirichlet(V, 1.0), K)'
		beta_old = beta
		beta_temp = zeros(K, V)
		gamma = [ones(K) for _ in 1:M]
		Elogtheta = [Base.MathConstants.eulergamma * ones(K) .- digamma(K) for _ in 1:M]
		Elogtheta_old = copy(Elogtheta)
		phi = ones(K, N[1]) / K
		elbo = 0
	
		model = new(K, M, V, N, C, copy(corp), topics, alpha, beta, eta_old, beta_temp, gamma, Elogtheta, Elogtheta_old, phi, elbo)
		
		for d in 1:model.M
			model.phi = ones(K, N[d]) / K
			model.elbo += Elogptheta(model) + Elogpz(model, d) + Elogpw(model, d) - Elogqtheta(model, d) - Elogqz(model, d)
		end

		return model
	end
end

function Elogptheta(model::LDA)
	"Compute the numerical value for E[log(P(theta))]."

	x = lgamma(sum(model.alpha)) - sum(lgamma.(model.alpha)) + dot(model.alpha - 1, model.Elogtheta[d])
	return x
end

function Elogpz(model::LDA, d::Int)
	"Compute the numerical value for E[log(P(z))]."

	counts = model.corp[d].counts
	x = dot(model.phi * counts, model.Elogtheta[d])
	return x
end

function Elogpw(model::LDA, d::Int)
	"Compute the numerical value for E[log(P(w))]."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	x = sum(model.phi .* log.(@boink model.beta[:,terms]) * counts)
	return x
end

function Elogqtheta(model::LDA, d::Int)
	"Compute the numerical value for E[log(q(theta))]."

	x = -entropy(Dirichlet(model.gamma[d]))
	return x
end

function Elogqz(model::LDA, d::Int)
	"Compute the numerical value for E[log(q(z))]."

	counts = model.corp[d].counts
	x = -sum([c * entropy(Categorical(model.phi[:,n])) for (n, c) in enumerate(counts)])
	return x
end

function update_elbo!(model::LDA, d::Int)
	"Update the evidence lower bound."

	model.elbo = 0
	for d in 1:model.M
		terms = model.corp[d].terms
		model.phi = model.beta_old[:,terms] .* exp.(model.Elogtheta_old[d])
		model.phi ./= sum(model.phi, 1)
		model.elbo += Elogptheta(model) + Elogpz(model, d) + Elogpw(model, d) - Elogqtheta(model, d) - Elogqz(model, d)
	end

	return model.elbo
end

function update_alpha!(model::LDA, niter::Integer, ntol::Real)
	"Update alpha."
	"Interior-point Newton method with log-barrier and back-tracking line search."

	Elogtheta_sum = sum([model.Elogtheta[d] for d in 1:model.M])

	nu = model.K
	for _ in 1:niter
		rho = 1.0
		alpha_grad = [nu / model.alpha[i] + model.M * (digamma(sum(model.alpha)) - digamma(model.alpha[i])) for i in 1:model.K] + Elogtheta_sum
		alpha_invhess_diag = -1 ./ (model.M * trigamma.(model.alpha) + nu ./ model.alpha.^2)
		p = (alpha_grad - dot(alpha_grad, alpha_invhess_diag) / (1 / (model.M * trigamma(sum(model.alpha))) + sum(alpha_invhess_diag))) .* alpha_invhess_diag
		
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
end

function update_beta!(model::LDA)
	"Reset beta variables."

	model.beta_old = model.beta
	model.beta = model.beta_temp ./ sum(model.beta_temp, 2)
	model.beta_temp = zeros(model.K, model.V)
end

function update_beta!(model::LDA, d::Int)
	"Update beta"
	"Analytic."

	terms, counts = model.corp[d].terms, model.corp[d].counts
	model.beta_temp[:,terms] += model.phi .* counts'		
end

function update_gamma!(model::LDA, d::Int)
	"Update gamma."
	"Analytic."

	counts = model.corp[d].counts
	@bumper model.gamma[d] = model.alpha + model.phi * counts	
end

function update_phi!(model::LDA, d::Int)
	"Update phi."
	"Analytic."

	terms = model.corp[d].terms
	model.phi = model.beta[:,terms] .* exp.(model.Elogtheta[d])
	model.phi ./= sum(model.phi, 1)
end

function update_Elogtheta!(model::LDA, d::Int)
	model.Elogtheta_old[d] = model.Elogtheta[d]
	model.Elogtheta[d] = digamma.(model.gamma[d]) - digamma(sum(model.gamma[d]))
end

function train!(model::LDA; iter::Integer=150, tol::Real=1.0, niter::Integer=1000, ntol::Real=1/model.K^2, viter::Integer=10, vtol::Real=1/model.K^2, chkelbo::Integer=1)
	"Coordinate ascent optimization procedure for latent Dirichlet allocation variational Bayes algorithm."

	@assert all(.!isnegative.([tol, ntol, vtol]))
	@assert all(ispositive.([iter, niter, viter, chkelbo]))

	for k in 1:iter
		chk = (k % chkelbo == 0)
		for d in 1:model.M	
			for _ in 1:viter
				update_phi!(model, d)
				update_gamma!(model, d)
				update_Elogtheta!(model, d)
				if norm(model.Elogtheta[d] - model.Elogtheta_old[d]) < vtol
					break
				end
			end
			update_beta!(model, d)
		end
		update_alpha!(model, niter, ntol)
		update_beta!(model)
		chk && update_elbo!(model)
		
		if checkELBO!(model, k, chk, tol)
			break
		end
	end

	model.topics = [reverse(sortperm(vec(model.beta[i,:]))) for i in 1:model.K]
	nothing
end















