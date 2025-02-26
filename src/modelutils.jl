struct TopicModelError <: Exception
    msg::String
end

Base.showerror(io::IO, e::TopicModelError) = print(io, "TopicModelError: ", e.msg)

"    showdocs(model::TopicModel, docs::Vector{Document})"
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)
showdocs(model::TopicModel, doc_indices::Vector{<:Integer}) = showdocs(model.corp, doc_indices)
showdocs(model::TopicModel, d::Integer) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc_range::UnitRange{<:Integer}) = showdocs(model.corp, collect(doc_range))
showdocs(model::TopicModel) = showdocs(model.corp)

"    showtitles(model::TopicModel, docs::Vector{Document})"
showtitles(model::TopicModel, docs::Vector{Document}) = showtitles(model.corp, docs)
showtitles(model::TopicModel, doc::Document) = showtitles(model.corp, doc)
showtitles(model::TopicModel, doc_indices::Vector{<:Integer}) = showtitles(model.corp, doc_indices)
showtitles(model::TopicModel, d::Integer) = showtitles(model.corp, d)
showtitles(model::TopicModel, doc_range::UnitRange{<:Integer}) = showtitles(model.corp, collect(doc_range))
showtitles(model::TopicModel) = showtitles(model.corp)

"    getvocab(model::TopicModel)"
getvocab(model::TopicModel) = getvocab(model.corp)

"    getusers(model::TopicModel)"
getusers(model::TopicModel) = getusers(model.corp)

## Display output for TopicModel objects.
Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::fLDA) = print(io, "Filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::CTM) = print(io, "Correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::fCTM) = print(io, "Filtered correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")
Base.show(io::IO, model::gpuLDA) = print(io, "GPU accelerated latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTM) = print(io, "GPU accelerated correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTPF) = print(io, "GPU accelerated collaborative topic Poisson factorization model with $(model.K) topics.")

function check_model(model::LDA)
	check_corp(model.corp) 
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))				|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))											|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])						|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])						|| throw(TopicModelError("C must contain sums of document counts."))
	isequal(length(model.alpha), model.K)											|| throw(TopicModelError("alpha must be of length K."))
	all(isfinite.(model.alpha))														|| throw(TopicModelError("alpha must be finite."))
	all(model.alpha .> 0)															|| throw(TopicModelError("alpha must be positive."))
	isequal(size(model.beta), (model.K, model.V))									|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))						|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(size(model.beta_old), (model.K, model.V))								|| throw(TopicModelError("beta_old must be of size (K, V)."))
	(isstochastic(model.beta_old, dims=2) | isempty(model.beta_old))				|| throw(TopicModelError("beta_old must be a right stochastic matrix."))
	isequal(model.beta_temp, zeros(model.K, model.V))								|| throw(TopicModelError("beta_temp must be a zero matrix of size (K, V)."))
	isequal(length(model.Elogtheta), model.M)										|| throw(TopicModelError("Elogtheta must be of length M."))
	all(Bool[isequal(length(model.Elogtheta[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("Elogtheta must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta[d])) for d in 1:model.M])				|| throw(TopicModelError("Elogtheta must be finite."))
	all(Bool[all(model.Elogtheta[d] .<= 0) for d in 1:model.M])						|| throw(TopicModelError("Elogtheta must be nonpositive."))
	isequal(length(model.Elogtheta_old), model.M)									|| throw(TopicModelError("Elogtheta_old must be of length M."))
	all(Bool[isequal(length(model.Elogtheta_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("Elogtheta_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta_old[d])) for d in 1:model.M])			|| throw(TopicModelError("Elogtheta_old must be finite."))
	all(Bool[all(model.Elogtheta_old[d] .<= 0) for d in 1:model.M])					|| throw(TopicModelError("Elogtheta_old must be nonpositive."))
	isequal(length(model.gamma), model.M)											|| throw(TopicModelError("gamma must be of length M."))
	all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])			|| throw(TopicModelError("gamma must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])					|| throw(TopicModelError("gamma must be finite."))
	all(Bool[all(model.gamma[d] .> 0) for d in 1:model.M])							|| throw(TopicModelError("gamma must be positive."))
	isfinite(model.elbo)															|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::fLDA)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))				|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))											|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])						|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])						|| throw(TopicModelError("C must contain sums of document counts."))
	(0 <= model.eta <= 1)															|| throw(TopicModelError("eta must belong to the interval [0,1]."))
	isequal(length(model.alpha), model.K)											|| throw(TopicModelError("alpha must be of length K."))
	all(isfinite.(model.alpha))														|| throw(TopicModelError("alpha must be finite."))
	all(model.alpha .> 0)															|| throw(TopicModelError("alpha must be positive."))
	isequal(length(model.kappa), model.V)											|| throw(TopicModelError("kappa must be of length V"))
	(isprobvec(model.kappa) | isempty(model.kappa))									|| throw(TopicModelError("kappa must be a probability vector."))
	isequal(length(model.kappa_old), model.V)										|| throw(TopicModelError("kappa_old must be of length V."))
	(isprobvec(model.kappa_old) | isempty(model.kappa_old))							|| throw(TopicModelError("kappa_old must be a probability vector."))
	isequal(model.kappa_temp, zeros(model.V))										|| throw(TopicModelError("kappa_temp must be a zero vector of length V."))
	isequal(size(model.beta), (model.K, model.V))									|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))						|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(size(model.beta_old), (model.K, model.V))								|| throw(TopicModelError("beta_old must be of size (K, V)."))
	(isstochastic(model.beta_old, dims=2) | isempty(model.beta_old))				|| throw(TopicModelError("beta_old must be a right stochastic matrix."))
	isequal(model.beta_temp, zeros(model.K, model.V))								|| throw(TopicModelError("beta_temp must be a zero matrix of size (K, V)."))
	isequal(length(model.Elogtheta), model.M)										|| throw(TopicModelError("Elogtheta must be of length M."))
	all(Bool[isequal(length(model.Elogtheta[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("Elogtheta must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta[d])) for d in 1:model.M])				|| throw(TopicModelError("Elogtheta must be finite."))
	all(Bool[all(model.Elogtheta[d] .<= 0) for d in 1:model.M])						|| throw(TopicModelError("Elogtheta must be nonpositive."))
	isequal(length(model.Elogtheta_old), model.M)									|| throw(TopicModelError("Elogtheta_old must be of length M."))
	all(Bool[isequal(length(model.Elogtheta_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("Elogtheta_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta_old[d])) for d in 1:model.M])			|| throw(TopicModelError("Elogtheta_old must be finite."))
	all(Bool[all(model.Elogtheta_old[d] .<= 0) for d in 1:model.M])					|| throw(TopicModelError("Elogtheta_old must be nonpositive."))
	isequal(length(model.gamma), model.M)											|| throw(TopicModelError("gamma must be of length M."))
	all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])			|| throw(TopicModelError("gamma must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])					|| throw(TopicModelError("gamma must be finite."))
	all(Bool[all(model.gamma[d] .> 0) for d in 1:model.M])							|| throw(TopicModelError("gamma must be positive."))
	isequal(length(model.tau), model.M)												|| throw(TopicModelError("tau must be of length M."))
	all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])			|| throw(TopicModelError("tau must contain vectors of lengths N."))
	all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])						|| throw(TopicModelError("tau must belong to the interval [0,1]."))
	isfinite(model.elbo)															|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::CTM)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))	
	isequal(model.M, length(model.corp))										|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])					|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])					|| throw(TopicModelError("C must contain sums of document counts."))	
	all(isfinite.(model.mu))													|| throw(TopicModelError("mu must be finite."))
	isequal(size(model.sigma), (model.K, model.K))								|| throw(TopicModelError("sigma must be of size (K, K)."))
	isposdef(model.sigma)														|| throw(TopicModelError("sigma must be positive-definite."))
	isequal(size(model.invsigma), (model.K, model.K))							|| throw(TopicModelError("invsigma must be of size (K, K)."))
	isposdef(model.invsigma)													|| throw(TopicModelError("invsigma must be positive-definite."))
	isequal(size(model.beta), (model.K, model.V))								|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))					|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(size(model.beta_old), (model.K, model.V))							|| throw(TopicModelError("beta_old must be of size (K, V)."))
	(isstochastic(model.beta_old, dims=2) | isempty(model.beta_old))			|| throw(TopicModelError("beta_old must be a right stochastic matrix."))
	isequal(model.beta_temp, zeros(model.K, model.V))							|| throw(TopicModelError("beta_temp must be a zero matrix of size (K, V)."))
	isequal(length(model.lambda), model.M)										|| throw(TopicModelError("lambda must be of length M."))
	all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("lambda must contain vectors of length K."))
	all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])				|| throw(TopicModelError("lambda must be finite."))
	isequal(length(model.lambda_old), model.M)									|| throw(TopicModelError("lambda_old must be of length M."))
	all(Bool[isequal(length(model.lambda_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("lambda_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.lambda_old[d])) for d in 1:model.M])			|| throw(TopicModelError("lambda_old must be finite."))
	isequal(length(model.vsq), model.M)											|| throw(TopicModelError("vsq must be of length M."))
	all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("vsq must contain vectors of length K."))
	all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])					|| throw(TopicModelError("vsq must be finite."))
	all(Bool[all(model.vsq[d] .> 0) for d in 1:model.M])						|| throw(TopicModelError("vsq must be positive."))
	isequal(length(model.logzeta), model.M)										|| throw(TopicModelError("logzeta must be of length M."))
	all(isfinite.(model.logzeta))												|| throw(TopicModelError("logzeta must be finite."))
	isfinite(model.elbo)														|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::fCTM)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))										|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])					|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])					|| throw(TopicModelError("C must contain sums of document counts."))
	(0 <= model.eta <= 1)														|| throw(TopicModelError("eta must belong to the interval [0,1]."))
	all(isfinite.(model.mu))													|| throw(TopicModelError("mu must be finite."))
	isequal(size(model.sigma), (model.K, model.K))								|| throw(TopicModelError("sigma must be of size (K, K)."))
	isposdef(model.sigma)														|| throw(TopicModelError("sigma must be positive-definite."))
	isequal(size(model.invsigma), (model.K, model.K))							|| throw(TopicModelError("invsigma must be of size (K, K)."))
	isposdef(model.invsigma)													|| throw(TopicModelError("invsigma must be positive-definite."))
	isequal(length(model.kappa), model.V)										|| throw(TopicModelError("kappa must be of length V."))
	(isprobvec(model.kappa) | isempty(model.kappa))								|| throw(TopicModelError("kappa must be a probability vector."))
	isequal(length(model.kappa_old), model.V)									|| throw(TopicModelError("kappa_old must be of length V."))
	(isprobvec(model.kappa_old) | isempty(model.kappa_old))						|| throw(TopicModelError("kappa_old must be a probability vector."))
	isequal(model.kappa_temp, zeros(model.V))									|| throw(TopicModelError("kappa_temp must be zero vector of length V."))
	isequal(size(model.beta), (model.K, model.V))								|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))					|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(size(model.beta_old), (model.K, model.V))							|| throw(TopicModelError("beta_old must be of size (K, V)."))
	(isstochastic(model.beta_old, dims=2) | isempty(model.beta_old))			|| throw(TopicModelError("beta_old must be a right stochastic matrix."))
	isequal(model.beta_temp, zeros(model.K, model.V))							|| throw(TopicModelError("beta_temp must be a zero matrix of size (K, V)."))
	isequal(length(model.lambda), model.M)										|| throw(TopicModelError("lambda must be of length M."))
	all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("lambda must contain vectors of length K."))
	all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])				|| throw(TopicModelError("lambda must be finite."))
	isequal(length(model.lambda_old), model.M)									|| throw(TopicModelError("lambda_old must be of length M."))
	all(Bool[isequal(length(model.lambda_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("lambda_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.lambda_old[d])) for d in 1:model.M])			|| throw(TopicModelError("lambda_old must be finite."))
	isequal(length(model.vsq), model.M)											|| throw(TopicModelError("vsq must be of length M."))
	all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("vsq must contain vectors of length K."))
	all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])					|| throw(TopicModelError("vsq must be finite."))
	all(Bool[all(model.vsq[d] .> 0) for d in 1:model.M])						|| throw(TopicModelError("vsq must be positive."))
	isequal(length(model.logzeta), model.M)										|| throw(TopicModelError("logzeta must be of length M."))
	all(isfinite.(model.logzeta))												|| throw(TopicModelError("logzeta must be finite."))
	isequal(length(model.tau), model.M)											|| throw(TopicModelError("tau must be of length M."))
	all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])		|| throw(TopicModelError("tau must contain vectors of lengths N."))
	all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])					|| throw(TopicModelError("tau must belong to the interval [0,1]."))
	isfinite(model.elbo)														|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::CTPF)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))			|| throw(TopicModelError("corpus users keys must form unit range of length U."))
	isequal(model.M, length(model.corp))										|| throw(TopicModelError("M must be equal to the number of documents in the corpus."))
	isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])			|| throw(TopicModelError("N must contain document lengths"))
	isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])			|| throw(TopicModelError("C must contain sums of document counts."))
	isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])		|| throw(TopicModelError("R must contain numbers of readers in documents."))
	model.a > 0																	|| throw(TopicModelError("a must be positive."))
	model.b > 0																	|| throw(TopicModelError("b must be positive."))
	model.c > 0																	|| throw(TopicModelError("c must be positive."))
	model.d > 0																	|| throw(TopicModelError("d must be positive."))
	model.e > 0																	|| throw(TopicModelError("e must be positive."))
	model.f > 0																	|| throw(TopicModelError("f must be positive."))
	model.g > 0																	|| throw(TopicModelError("g must be positive."))
	model.h > 0																	|| throw(TopicModelError("h must be positive."))
	isequal(size(model.alef), (model.K, model.V))								|| throw(TopicModelError("alef must be of size (K, V)."))
	all(isfinite.(model.alef))													|| throw(TopicModelError("alef must be finite."))
	all(model.alef .> 0)														|| throw(TopicModelError("alef must be positive."))
	isequal(size(model.alef_old), (model.K, model.V))							|| throw(TopicModelError("alef_old must be of size (K, V)."))
	all(isfinite.(model.alef_old))												|| throw(TopicModelError("alef_old must be finite."))
	all(model.alef_old .> 0)													|| throw(TopicModelError("alef_old must be positive."))
	isequal(model.alef_temp, fill(model.a, model.K, model.V))					|| throw(TopicModelError("alef_temp must be a fill a matrix of size (K, V)."))
	isequal(length(model.bet), model.K)											|| throw(TopicModelError("bet must be of length K"))
	all(isfinite.(model.bet))													|| throw(TopicModelError("bet must be finite."))
	all(model.bet .> 0)															|| throw(TopicModelError("bet must be positive."))
	isequal(length(model.bet_old), model.K)										|| throw(TopicModelError("bet_old must be of length K"))
	all(isfinite.(model.bet_old))												|| throw(TopicModelError("bet_old must be finite."))
	all(model.bet_old .> 0)														|| throw(TopicModelError("bet_old must be positive."))
	isequal(length(model.gimel), model.M)										|| throw(TopicModelError("gimel must be of length M."))
	all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("gimel must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gimel[d])) for d in 1:model.M])				|| throw(TopicModelError("gimel must be finite."))
	all(Bool[all(model.gimel[d] .> 0) for d in 1:model.M])						|| throw(TopicModelError("gimel must be positive."))
	isequal(length(model.gimel_old), model.M)									|| throw(TopicModelError("gimel_old must be of length M."))
	all(Bool[isequal(length(model.gimel_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("gimel_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gimel_old[d])) for d in 1:model.M])			|| throw(TopicModelError("gimel_old must be finite."))
	all(Bool[all(model.gimel_old[d] .> 0) for d in 1:model.M])					|| throw(TopicModelError("gimel_old must be positive."))
	isequal(length(model.dalet), model.K)										|| throw(TopicModelError("dalet must be of length K."))
	all(isfinite.(model.dalet))													|| throw(TopicModelError("dalet must be finite."))
	all(model.dalet .> 0)														|| throw(TopicModelError("dalet must be positive."))
	isequal(length(model.dalet_old), model.K)									|| throw(TopicModelError("dalet_old must be of length K."))
	all(isfinite.(model.dalet_old))												|| throw(TopicModelError("dalet_old must be finite."))
	all(model.dalet_old .> 0)													|| throw(TopicModelError("dalet_old must be positive."))
	isequal(size(model.he), (model.K, model.U))									|| throw(TopicModelError("he must be of size (K, U)"))
	all(isfinite.(model.he))													|| throw(TopicModelError("he must be finite."))
	all(model.he .> 0)															|| throw(TopicModelError("he must be positive."))
	isequal(size(model.he_old), (model.K, model.U))								|| throw(TopicModelError("he_old must be of size (K, U)"))
	all(isfinite.(model.he_old))												|| throw(TopicModelError("he_old must be finite."))
	all(model.he_old .> 0)														|| throw(TopicModelError("he_old must be positive."))
	isequal(model.he_temp, fill(model.e, model.K, model.U))						|| throw(TopicModelError("he_temp must be a fill e matrix of size (K, U)."))
	isequal(length(model.vav), model.K)											|| throw(TopicModelError("vav must be of length K."))
	all(isfinite.(model.vav))													|| throw(TopicModelError("vav must be finite."))
	all(model.vav .> 0)															|| throw(TopicModelError("vav must be positive."))
	isequal(length(model.vav_old), model.K)										|| throw(TopicModelError("vav_old must be of length K."))
	all(isfinite.(model.vav_old))												|| throw(TopicModelError("vav_old must be finite."))
	all(model.vav_old .> 0)														|| throw(TopicModelError("vav_old must be positive."))
	isequal(length(model.zayin), model.M)										|| throw(TopicModelError("zayin must be of length M."))
	all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])		|| throw(TopicModelError("zayin must contain vectors of length K."))
	all(Bool[all(isfinite.(model.zayin[d])) for d in 1:model.M])				|| throw(TopicModelError("zayin must be finite."))
	all(Bool[all(model.zayin[d] .> 0) for d in 1:model.M])						|| throw(TopicModelError("zayin must be positive."))
	isequal(length(model.zayin_old), model.M)									|| throw(TopicModelError("zayin_old must be of length M."))
	all(Bool[isequal(length(model.zayin_old[d]), model.K) for d in 1:model.M])	|| throw(TopicModelError("zayin_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.zayin_old[d])) for d in 1:model.M])			|| throw(TopicModelError("zayin_old must be finite."))
	all(Bool[all(model.zayin_old[d] .> 0) for d in 1:model.M])					|| throw(TopicModelError("zayin_old must be positive."))
	isequal(length(model.het), model.K)											|| throw(TopicModelError("het must be of length K."))
	all(isfinite.(model.het))													|| throw(TopicModelError("het must be finite."))
	all(model.het .> 0)															|| throw(TopicModelError("het must be positive."))
	isequal(length(model.het_old), model.K)										|| throw(TopicModelError("het_old must be of length K."))
	all(isfinite.(model.het_old))												|| throw(TopicModelError("het_old must be finite."))
	all(model.het_old .> 0)														|| throw(TopicModelError("het_old must be positive."))
	isfinite(model.elbo)														|| throw(TopicModelError("elbo must be finite."))
	nothing	
end

function check_model(model::gpuLDA)
	check_corp(model.corp) 
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))							|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))														|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])									|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])									|| throw(TopicModelError("C must contain sums of document counts."))
	isequal(length(model.alpha), model.K)														|| throw(TopicModelError("alpha must be of length K."))
	all(isfinite.(model.alpha))																	|| throw(TopicModelError("alpha must be finite."))
	all(model.alpha .> 0)																		|| throw(TopicModelError("alpha must be positive."))
	isequal(size(model.beta), (model.K, model.V))												|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))									|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(length(model.Elogtheta), model.M)													|| throw(TopicModelError("Elogtheta must be of length M."))
	all(Bool[isequal(length(model.Elogtheta[d]), model.K) for d in 1:model.M])					|| throw(TopicModelError("Elogtheta must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta[d])) for d in 1:model.M])							|| throw(TopicModelError("Elogtheta must be finite."))
	all(Bool[all(model.Elogtheta[d] .<= 0) for d in 1:model.M])									|| throw(TopicModelError("Elogtheta must be nonpositive."))
	isequal(length(model.gamma), model.M)														|| throw(TopicModelError("gamma must be of length M."))
	all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])						|| throw(TopicModelError("gamma must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])								|| throw(TopicModelError("gamma must be finite."))
	all(Bool[all(model.gamma[d] .> 0) for d in 1:model.M])										|| throw(TopicModelError("gamma must be positive."))
	isequal(length(model.phi), model.M)															|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])			|| throw(TopicModelError("phi must contain matrices of size (K, N)."))
	all(Bool[isstochastic(model.phi[d], dims=1) | isempty(model.phi[d]) for d in 1:model.M])	|| throw(TopicModelError("phi must contain left stochastic matrices."))
	isfinite(model.elbo)																		|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::gpuCTM)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))							|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))	
	isequal(model.M, length(model.corp))														|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])									|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])									|| throw(TopicModelError("C must contain sums of document counts."))	
	all(isfinite.(model.mu))																	|| throw(TopicModelError("mu must be finite."))
	isequal(size(model.sigma), (model.K, model.K))												|| throw(TopicModelError("sigma must be of size (K, K)."))
	isposdef(model.sigma)																		|| throw(TopicModelError("sigma must be positive-definite."))
	isequal(size(model.invsigma), (model.K, model.K))											|| throw(TopicModelError("invsigma must be of size (K, K)."))
	isposdef(model.invsigma)																	|| throw(TopicModelError("invsigma must be positive-definite."))
	isequal(size(model.beta), (model.K, model.V))												|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))									|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(length(model.lambda), model.M)														|| throw(TopicModelError("lambda must be of length M."))
	all(Bool[isequal(length(model.lambda[d]), model.K) for d in 1:model.M])						|| throw(TopicModelError("lambda must contain vectors of length K."))
	all(Bool[all(isfinite.(model.lambda[d])) for d in 1:model.M])								|| throw(TopicModelError("lambda must be finite."))
	isequal(length(model.vsq), model.M)															|| throw(TopicModelError("vsq must be of length M."))
	all(Bool[isequal(length(model.vsq[d]), model.K) for d in 1:model.M])						|| throw(TopicModelError("vsq must contain vectors of length K."))
	all(Bool[all(isfinite.(model.vsq[d])) for d in 1:model.M])									|| throw(TopicModelError("vsq must be finite."))
	all(Bool[all(model.vsq[d] .> 0) for d in 1:model.M])										|| throw(TopicModelError("vsq must be positive."))
	isequal(length(model.logzeta), model.M)														|| throw(TopicModelError("logzeta must be of length M."))
	all(isfinite.(model.logzeta))																|| throw(TopicModelError("logzeta must be finite."))
	isequal(length(model.phi), model.M)															|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])			|| throw(TopicModelError("phi must contain matrices of size (K, N)."))
	all(Bool[isstochastic(model.phi[d], dims=1) | isempty(model.phi[d]) for d in 1:model.M])	|| throw(TopicModelError("phi must contain left stochastic matrices."))
	isfinite(model.elbo)																		|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::gpuCTPF)
	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))					|| throw(TopicModelError("corpus vocab keys must form unit range of length V."))
	isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))					|| throw(TopicModelError("corpus users keys must form unit range of length U."))
	isequal(model.M, length(model.corp))												|| throw(TopicModelError("M must be equal to the number of documents in the corpus."))
	isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])					|| throw(TopicModelError("N must contain document lengths"))
	isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])					|| throw(TopicModelError("C must contain sums of document counts."))
	isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])				|| throw(TopicModelError("R must contain numbers of readers in documents."))
	model.a > 0																			|| throw(TopicModelError("a must be positive."))
	model.b > 0																			|| throw(TopicModelError("b must be positive."))
	model.c > 0																			|| throw(TopicModelError("c must be positive."))
	model.d > 0																			|| throw(TopicModelError("d must be positive."))
	model.e > 0																			|| throw(TopicModelError("e must be positive."))
	model.f > 0																			|| throw(TopicModelError("f must be positive."))
	model.g > 0																			|| throw(TopicModelError("g must be positive."))
	model.h > 0																			|| throw(TopicModelError("h must be positive."))
	isequal(size(model.alef), (model.K, model.V))										|| throw(TopicModelError("alef must be of size (K, V)."))
	all(isfinite.(model.alef))															|| throw(TopicModelError("alef must be finite."))
	all(model.alef .> 0)																|| throw(TopicModelError("alef must be positive."))
	isequal(length(model.bet), model.K)													|| throw(TopicModelError("bet must be of length K"))
	all(isfinite.(model.bet))															|| throw(TopicModelError("bet must be finite."))
	all(model.bet .> 0)																	|| throw(TopicModelError("bet must be positive."))														
	isequal(length(model.gimel), model.M)												|| throw(TopicModelError("gimel must be of length M."))
	all(Bool[isequal(length(model.gimel[d]), model.K) for d in 1:model.M])				|| throw(TopicModelError("gimel must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gimel[d])) for d in 1:model.M])						|| throw(TopicModelError("gimel must be finite."))
	all(Bool[all(model.gimel[d] .> 0) for d in 1:model.M])								|| throw(TopicModelError("gimel must be positive."))
	isequal(length(model.dalet), model.K)												|| throw(TopicModelError("dalet must be of length K."))
	all(isfinite.(model.dalet))															|| throw(TopicModelError("dalet must be finite."))
	all(model.dalet .> 0)																|| throw(TopicModelError("dalet must be positive."))
	isequal(size(model.he), (model.K, model.U))											|| throw(TopicModelError("he must be of size (K, U)"))
	all(isfinite.(model.he))															|| throw(TopicModelError("he must be finite."))
	all(model.he .> 0)																	|| throw(TopicModelError("he must be positive."))
	isequal(length(model.vav), model.K)													|| throw(TopicModelError("vav must be of length K."))
	all(isfinite.(model.vav))															|| throw(TopicModelError("vav must be finite."))
	all(model.vav .> 0)																	|| throw(TopicModelError("vav must be positive."))
	isequal(length(model.zayin), model.M)												|| throw(TopicModelError("zayin must be of length M."))
	all(Bool[isequal(length(model.zayin[d]), model.K) for d in 1:model.M])				|| throw(TopicModelError("zayin must contain vectors of length K."))
	all(Bool[all(isfinite.(model.zayin[d])) for d in 1:model.M])						|| throw(TopicModelError("zayin must be finite."))
	all(Bool[all(model.zayin[d] .> 0) for d in 1:model.M])								|| throw(TopicModelError("zayin must be positive."))
	isequal(length(model.het), model.K)													|| throw(TopicModelError("het must be of length K."))
	all(isfinite.(model.het))															|| throw(TopicModelError("het must be finite."))
	all(model.het .> 0)																	|| throw(TopicModelError("het must be positive."))
	isequal(length(model.phi), model.M)													|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])	|| throw(TopicModelError("phi must contain matrices of size (K, N)."))
	all(Bool[isstochastic(model.phi[d], dims=1) for d in 1:model.M])					|| throw(TopicModelError("phi must contain left stochastic matrices."))
	isequal(length(model.xi), model.M)													|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in 1:model.M])	|| throw(TopicModelError("xi must contain matrices of size (2K, R)."))
	all(Bool[isstochastic(model.xi[d], dims=1) for d in 1:model.M])						|| throw(TopicModelError("xi must contain left stochastic matrices."))
	isfinite(model.elbo)																|| throw(TopicModelError("elbo must be finite"))
	nothing	
end

"""
    check_model(model::TopicModel)

Check model parameters.
"""
check_model(model::TopicModel) = check_model(model)

## Update gpuLDA model data in GPU RAM.
function update_buffer!(model::gpuLDA)
	terms = vcat([doc.terms for doc in model.corp]...) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp]...)
		
	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_cumsum = cumsum([0; model.N])
	J_cumsum = cumsum([0; J])

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	model.N_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_cumsum)
	model.J_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_cumsum)

	model.alpha_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alpha)
	model.beta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.Elogtheta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.Elogtheta...))
	model.Elogtheta_sum_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=zeros(Float32, model.K))
	model.Elogtheta_dist_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=zeros(Float32, model.M))
	model.gamma_buffer = cl.Buffer(Float32, model.context, :rw, model.K * model.M)
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * sum(model.N))
end

## Update gpuCTM model data in GPU RAM.
function update_buffer!(model::gpuCTM)
	terms = vcat([doc.terms for doc in model.corp]...) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp]...)

	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_cumsum = cumsum([0; model.N])
	J_cumsum = cumsum([0; J])

	model.C_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=model.C)
	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	model.N_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_cumsum)
	model.J_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_cumsum)

	model.sigma_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=Matrix(model.sigma))
	model.invsigma_buffer = cl.Buffer(Float32, model.context, (:r, :copy), hostbuf=Matrix(model.invsigma))
	model.mu_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.mu)
	model.beta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.lambda_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.lambda...))
	model.lambda_dist_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=zeros(Float32, model.M))
	model.vsq_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.vsq...))
	model.logzeta_buffer = cl.Buffer(Float32, model.context, :rw, model.M)
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * sum(model.N))
	model.phi_count_buffer = cl.Buffer(Float32, model.context, :rw, model.K * model.M)

	cl.set_arg!(model.lambda_kernel, 15, cl.LocalMem(Float32, model.K))
	cl.set_arg!(model.lambda_kernel, 16, cl.LocalMem(Float32, model.K))
	cl.set_arg!(model.lambda_kernel, 17, cl.LocalMem(Float32, model.K^2))
end

## Update gpuCTPF model data in GPU RAM.
function update_buffer!(model::gpuCTPF)		
	terms = vcat([doc.terms for doc in model.corp]...) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp]...)

	readers = vcat([doc.readers for doc in model.corp]...) .- 1
	readers_sortperm = sortperm(readers) .- 1
	ratings = vcat([doc.ratings for doc in model.corp]...)
	

	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	Y = zeros(Int, model.U)
	for r in readers
		Y[r+1] += 1
	end

	N_cumsum = cumsum([0; model.N])
	J_cumsum = cumsum([0; J])
	R_cumsum = cumsum([0; model.R])
	Y_cumsum = cumsum([0; Y])

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)
	model.readers_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=[readers; 0])
	model.ratings_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=[ratings; 0])
	model.readers_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=[readers_sortperm; 0])
	model.N_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_cumsum)
	model.J_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_cumsum)
	model.R_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=R_cumsum)
	model.Y_cumsum_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=Y_cumsum)

	model.alef_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alef)
	model.bet_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.bet)
	model.vav_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.vav)
	model.gimel_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gimel...))
	model.zayin_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.zayin...))
	model.dalet_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.dalet)
	model.het_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.het)
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * sum(model.N))

	if model.U > 0
		model.he_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.he)
	else
		model.he_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=Float32[0])
	end

	if sum(model.R) > 0
		model.xi_buffer = cl.Buffer(Float32, model.context, :rw, 2 * model.K * sum(model.R))
	else
		model.xi_buffer = cl.Buffer(Float32, model.context, :rw, 1)
	end
end

function update_host!(model::TopicModel)
	nothing
end

## Update gpuLDA model data in CPU RAM.
function update_host!(model::gpuLDA)
	N_cumsum = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_cumsum[d+1] = N_cumsum[d] + model.N[d]
	end

	model.beta = reshape(cl.read(model.queue, model.beta_buffer), model.K, model.V)	
	Elogtheta_host = reshape(cl.read(model.queue, model.Elogtheta_buffer), model.K, model.M)
	model.Elogtheta = [Elogtheta_host[:,d] for d in 1:model.M]
	model.Elogtheta_sum = cl.read(model.queue, model.Elogtheta_sum_buffer)
	model.Elogtheta_dist = cl.read(model.queue, model.Elogtheta_dist_buffer)[1:model.M]
	gamma_host = reshape(cl.read(model.queue, model.gamma_buffer), model.K, model.M)
	model.gamma = [gamma_host[:,d] for d in 1:model.M]
	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N))
	model.phi = [phi_host[:,N_cumsum[d]+1:N_cumsum[d+1]] for d in 1:model.M]
end

## Update gpuCTM model data in CPU RAM.
function update_host!(model::gpuCTM)
	N_cumsum = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_cumsum[d+1] = N_cumsum[d] + model.N[d]
	end

	model.mu = cl.read(model.queue, model.mu_buffer)
	model.sigma = Symmetric(reshape(cl.read(model.queue, model.sigma_buffer), model.K, model.K))
	model.beta = reshape(cl.read(model.queue, model.beta_buffer), model.K, model.V)
	lambda_host = reshape(cl.read(model.queue, model.lambda_buffer), model.K, model.M)
	model.lambda = [lambda_host[:,d] for d in 1:model.M]
	model.lambda_dist = cl.read(model.queue,  model.lambda_dist_buffer)[1:model.M]
	vsq_host = reshape(cl.read(model.queue, model.vsq_buffer), model.K, model.M)
	model.vsq = [vsq_host[:,d] for d in 1:model.M]
	model.logzeta = cl.read(model.queue, model.logzeta_buffer)[1:model.M]
	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N))
	model.phi = [phi_host[:,N_cumsum[d]+1:N_cumsum[d+1]] for d in 1:model.M]
end

## Update gpuCTPF model data in CPU RAM.
function update_host!(model::gpuCTPF)
	N_cumsum = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_cumsum[d+1] = N_cumsum[d] + model.N[d]
	end

	R_cumsum = zeros(Int, model.M + 1)
	for d in 1:model.M
		R_cumsum[d+1] = R_cumsum[d] + model.R[d]
	end

	model.alef = reshape(cl.read(model.queue, model.alef_buffer), model.K, model.V)
	model.bet = cl.read(model.queue, model.bet_buffer)
	model.vav = cl.read(model.queue, model.vav_buffer)
	gimel_host = reshape(cl.read(model.queue, model.gimel_buffer), model.K, model.M)
	model.gimel = [gimel_host[:,d] for d in 1:model.M]
	zayin_host = reshape(cl.read(model.queue, model.zayin_buffer), model.K, model.M)
	model.zayin = [zayin_host[:,d] for d in 1:model.M]	
	model.dalet = cl.read(model.queue, model.dalet_buffer)
	model.het = cl.read(model.queue, model.het_buffer)
	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N))
	model.phi = [phi_host[:,N_cumsum[d]+1:N_cumsum[d+1]] for d in 1:model.M]

	if model.U > 0
		model.he = reshape(cl.read(model.queue, model.he_buffer), model.K, model.U)
	end

	if sum(model.R) > 0
		xi_host = reshape(cl.read(model.queue, model.xi_buffer), 2 * model.K, sum(model.R))
		model.xi = [xi_host[:,R_cumsum[d]+1:R_cumsum[d+1]] for d in 1:model.M]
	end
end

## Check and print value of delta_elbo.
## If abs(delta_elbo) < tol, terminate algorithm.
function check_elbo!(model::TopicModel, checkelbo::Real, printelbo::Bool, k::Int, tol::Real)
	if k % checkelbo == 0
		update_host!(model)
		delta_elbo = -(model.elbo - update_elbo!(model))
		printelbo && println(k, " ∆elbo: ", round(delta_elbo, digits=3))

		if delta_elbo < tol
			return true
		end
	end
	false
end

"""
    gendoc(model::Union{LDA, gpuLDA, fLDA}, laplace_smooth::Real=0.0)

Generate artificial document from LDA or gpuLDA generative model.

The `laplace_smooth` parameter governs the amount of Laplace smoothing applied to the topic-term distribution.
"""
function gendoc(model::Union{LDA, gpuLDA, fLDA}, laplace_smooth::Real=0.0)
	(laplace_smooth >= 0) || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	theta = rand(Dirichlet(model.alpha))
	
	topic_dist = Categorical(theta)
	vocab_dist = [Categorical((model.beta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	
	term_count = Dict{Int, Int}()
	for _ in 1:C
		z = rand(topic_dist)
		w = rand(vocab_dist[z])
		haskey(term_count, w) ? term_count[w] += 1 : term_count[w] = 1
	end

	doc = Document(terms=collect(keys(term_count)), counts=collect(values(term_count)))
	return doc
end

"    gendoc(model::Union{CTM, gpuCTM, fCTM}, laplace_smooth::Real=0.0)"
function gendoc(model::Union{CTM, gpuCTM, fCTM}, laplace_smooth::Real=0.0)
	(laplace_smooth >= 0) || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	theta = additive_logistic(rand(MvNormal(model.mu, model.sigma)))
	
	topic_dist = Categorical(theta)
	vocab_dist = [Categorical((model.beta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	
	term_count = Dict{Int, Int}()
	for _ in 1:C
		z = rand(topicdist)
		w = rand(vocab_dist[z])
		haskey(term_count, w) ? term_count[w] += 1 : term_count[w] = 1
	end

	doc = Document(terms=collect(keys(term_count)), counts=collect(values(term_count)))
	return doc
end

"""
    gencorp(model::TopicModel, M::Integer; laplace_smooth::Real=0.0)

Generate artificial corpus using specified generative model.

The `laplace_smooth` parameter governs the amount of Laplace smoothing applied to the topic-term distribution.
"""
function gencorp(model::TopicModel, M::Integer; laplace_smooth::Real=0.0)
	(M > 0)					|| throw(ArgumentError("corp_size parameter must be a positive integer."))
	(laplace_smooth >= 0)	|| throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	corp = Corpus(vocab=model.corp.vocab, users=model.corp.users)
	corp.docs = [gendoc(model, laplace_smooth) for d in 1:M]
	return corp
end

"""
    showtopics(model::TopicModel, V::Integer=15; topics::Union{<:Integer, Vector{<:Integer}, UnitRange{<:Integer}}=1:model.K, cols::Integer=4)

Display the top `T` terms for each topic.
"""
function showtopics(model::TopicModel, V::Integer=15; topics::Union{<:Integer, Vector{<:Integer}, UnitRange{<:Integer}}=1:model.K, cols::Integer=4)
	(V > 0)									|| throw(ArgumentError("number of displayed terms must be a positive integer."))
	checkbounds(Bool, 1:model.K, topics)	|| throw(ArgumentError("some topic indices are outside range."))
	(cols > 0)								|| throw(ArgumentError("cols must be a positive integer."))
	V = min(V, model.V)	
	cols = min(cols, length(topics))

	maxjspacings = [maximum([length(model.corp.vocab[j]) for j in topic[1:V]]) for topic in model.topics]
	topic_blocks = Iterators.partition(topics, cols)

	for (n, topic_block) in enumerate(topic_blocks)
		for j in 0:V
			for (k, i) in enumerate(topic_block)
				if j == 0
					jspacing = max(4, maxjspacings[i] - length("$i") - 2)
					k == cols ? print(Crayon(foreground=:yellow, bold=true), "topic $i") : print(Crayon(foreground=:yellow, bold=true), "topic $i" * " "^jspacing)
				else
					jspacing = max(6 + length("$i"), maxjspacings[i]) - length(model.corp.vocab[model.topics[i][j]]) + 4
					k == cols ? print(Crayon(foreground=:white, bold=false), model.corp.vocab[model.topics[i][j]]) : print(Crayon(foreground=:white, bold=false), model.corp.vocab[model.topics[i][j]] * " "^jspacing)
				end
			end
			println()
		end

		if n < length(topic_blocks)
			println()
		end
	end
end

"""
    showlibs(model::Union{CTPF, gpuCTPF}, users::Vector{<:Integer})

Display the documents in user libraries.
"""
function showlibs(model::Union{CTPF, gpuCTPF}, users::Vector{<:Integer})
	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("some user indices are outside range."))
	
	for (n, u) in enumerate(users)
		if isempty(model.libs[u])
			continue
		end

		@juliadots "User $u\n"
		try
			if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		
		catch
			@juliadots model.corp.users[u] * "\n"
		end
		
		for d in model.libs[u]
			print(Crayon(foreground=:yellow, bold=true), " • ")
			isempty(model.corp[d].title) ? println(Crayon(foreground=:white, bold=false), "Document $d") : println(Crayon(foreground=:white, bold=false), "$(model.corp[d].title)")
		end

		if n < length(users)
			println()
		end
	end
end

showlibs(model::Union{CTPF, gpuCTPF}, user::Integer) = showlibs(model, [user])
showlibs(model::Union{CTPF, gpuCTPF}, user_range::UnitRange{<:Integer}) = showlibs(model, collect(user_range))
showlibs(model::Union{CTPF, gpuCTPF}) = showlibs(model, 1:length(model.libs))

"""
    showdrecs(model::Union{CTPF, gpuCTPF}, docs::Vector{<:Integer}, U::Integer=15; cols::Integer=1)

Display the top `U` user recommendations for a set of documents.
"""
function showdrecs(model::Union{CTPF, gpuCTPF}, docs::Vector{<:Integer}, U::Integer=15; cols::Integer=1)
	checkbounds(Bool, 1:model.M, docs) 	|| throw(ArgumentError("some document indices are outside range."))
	(U > 0) 							|| throw(ArgumentError("number of displayed users must be a positive integer."))
	(cols > 0)							|| throw(ArgumentError("cols must be a positive integer."))
	U = min(U, model.U)

	for (n, d) in enumerate(docs)
		if isempty(model.drecs[d])
			continue
		end

		@juliadots "Document $d\n"
		if !isempty(model.corp[d].title)
			@juliadots model.corp[d].title * "\n"
		end

		usercols = collect(Iterators.partition(model.drecs[d][1:U], Int(ceil(U / cols))))
		rankcols = collect(Iterators.partition(1:U, Int(ceil(U / cols))))

		for i in 1:length(usercols[1])
			for j in 1:length(usercols)
				try
					uspacing = maximum([length(model.corp.users[u]) for u in usercols[j]]) - length(model.corp.users[usercols[j][i]]) + 4
					rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
					print(Crayon(foreground=:yellow, bold=true), string(rankcols[j][i]) * ". " * " "^rspacing)
					j == length(usercols) ? print(Crayon(foreground=:white, bold=false), model.corp.users[usercols[j][i]]) : print(Crayon(foreground=:white, bold=false), users[usercols[j][i]] * " "^uspacing)
				
				catch
					nothing
				end
			end
			println()
		end

		if n < length(docs)
			println()
		end
	end
end

showdrecs(model::Union{CTPF, gpuCTPF}, doc::Integer, U::Integer=15; cols::Integer=1) = showdrecs(model, [doc], U, cols=cols)
showdrecs(model::Union{CTPF, gpuCTPF}, docs::UnitRange{<:Integer}, U::Integer=15; cols::Integer=1) = showdrecs(model, collect(docs), U, cols=cols)

"""
    showurecs(model::Union{CTPF, gpuCTPF}, users::Vector{<:Integer}, M::Integer=15; cols::Integer=1)

Show the top `M` document recommendations for a set of users.
"""
function showurecs(model::Union{CTPF, gpuCTPF}, users::Vector{<:Integer}, M::Integer=15; cols::Integer=1)
	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("some user indices are outside range."))
	(M > 0) 							|| throw(ArgumentError("number of displayed documents must be a positive integer."))
	(cols > 0)							|| throw(ArgumentError("cols must be a positive integer."))
	M = min(M, model.M)

	for (n, u) in enumerate(users)
		if isempty(model.urecs[u])
			continue
		end

		@juliadots "User $u\n"
		try 
			if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		
		catch 
			@juliadots model.corp.users[u] * "\n"
		end

		docucols = collect(Iterators.partition(model.urecs[u][1:M], Int(ceil(M / cols))))
		rankcols = collect(Iterators.partition(1:M, Int(ceil(M / cols))))

		for i in 1:length(docucols[1])
			for j in 1:length(docucols)
				try
					!isempty(model.corp[docucols[j][i]].title) ? title = model.corp[docucols[j][i]].title : title = "Document $(docucols[j][i])"
					dspacing = maximum([max(4 + length("$(docucols[j][i])"), length(model.corp[d].title)) for d in docucols[j]]) - length(title) + 4
					rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
					print(Crayon(foreground=:yellow, bold=true), string(rankcols[j][i]) * ". " * " "^rspacing)
					j == length(docucols) ? print(Crayon(foreground=:white, bold=false), title) : print(Crayon(foreground=:white, bold=false), title * " "^dspacing)

				catch
					nothing
				end
			end
			println()
		end

		if n < length(users)
			println()
		end
	end
end

showurecs(model::Union{CTPF, gpuCTPF}, user::Integer, M::Integer=15; cols::Integer=1) = showurecs(model, [user], M, cols=cols)
showurecs(model::Union{CTPF, gpuCTPF}, users::UnitRange{<:Integer}, M::Integer=15; cols::Integer=1) = showurecs(model, collect(users), M, cols=cols)

"""
    predict(corp::Corpus, train_model::Union{LDA, gpuLDA}; iter::Integer=10, tol::Real=1/train_model.K^2)

Predict topic distributions for a corpus of documents based on a trained model.
"""
function predict(corp::Corpus, train_model::Union{LDA, gpuLDA}; iter::Integer=10, tol::Real=1/train_model.K^2)
	check_corp(corp)
	check_model(train_model)
	(corp.vocab == train_model.corp.vocab)	|| throw(CorpusError("predict corpus and train_model corpus must have identical vocabularies."))
	(tol .>= 0)								|| throw(ArgumentError("tolerance parameter must be nonnegative."))
	(iter .>= 0)							|| throw(ArgumentError("iteration parameter must be nonnegative."))

	model = LDA(corp, train_model.K)
	model.alpha = train_model.alpha
	model.beta = train_model.beta
	model.topics = train_model.topics

	for d in 1:model.M
		for v in 1:iter
			update_phi!(model, d)
			update_gamma!(model, d)
			update_Elogtheta!(model, d)
			if norm(model.Elogtheta[d] - model.Elogtheta_old[d]) < tol
				break
			end
		end
	end

	return model
end

"    predict(corp::Corpus, train_model::fLDA; iter::Integer=10, tol::Real=1/train_model.K^2)"
function predict(corp::Corpus, train_model::fLDA; iter::Integer=10, tol::Real=1/train_model.K^2)
	check_corp(corp)
	check_model(train_model)
	(corp.vocab == train_model.corp.vocab)	|| throw(CorpusError("predict corpus and train_model corpus must have identical vocabularies."))
	(tol .>= 0)								|| throw(ArgumentError("tolerance parameter must be nonnegative."))
	(iter .>= 0)							|| throw(ArgumentError("iteration parameter must be nonnegative."))

	model = fLDA(corp, train_model.K)
	model.alpha = train_model.alpha
	model.beta = train_model.beta
	model.topics = train_model.topics

	for d in 1:model.M
		for v in 1:iter
			update_phi!(model, d)
			update_tau!(model, d)
			update_gamma!(model, d)
			update_Elogtheta!(model, d)
			if norm(model.Elogtheta[d] - model.Elogtheta_old[d]) < vtol
				break
			end
		end
	end

	return model
end

"    predict(corp::Corpus, train_model::Union{CTM, gpuCTM}; iter::Integer=10, tol::Real=1/train_model.K^2, niter::Integer=1000, ntol::Real=1/train_model.K^2)"
function predict(corp::Corpus, train_model::Union{CTM, gpuCTM}; iter::Integer=10, tol::Real=1/train_model.K^2, niter::Integer=1000, ntol::Real=1/train_model.K^2)
	check_corp(corp)
	check_model(train_model)
	(corp.vocab == train_model.corp.vocab)	|| throw(CorpusError("predict corpus and train_model corpus must have identical vocabularies."))
	all([tol, ntol] .>= 0)					|| throw(ArgumentError("tolerance parameters must be nonnegative."))
	all([iter, niter] .>= 0)				|| throw(ArgumentError("iteration parameters must be nonnegative."))

	model = CTM(corp, train_model.K)
	model.mu = train_model.mu
	model.sigma = train_model.sigma
	model.invsigma = train_model.invsigma
	model.beta = train_model.beta
	model.topics = train_model.topics

	for d in 1:model.M
		for v in 1:iter
			update_phi!(model, d)
			update_logzeta!(model, d)
			update_vsq!(model, d, niter, ntol)
			update_lambda!(model, d, niter, ntol)
			if norm(model.lambda[d] - model.lambda_old[d]) < tol
				break
			end
		end
	end

	return model
end

"   predict(corp::Corpus, train_model::fCTM; iter::Integer=10, tol::Real=1/train_model.K^2, niter::Integer=1000, ntol::Real=1/train_model.K^2)"
function predict(corp::Corpus, train_model::fCTM; iter::Integer=10, tol::Real=1/train_model.K^2, niter::Integer=1000, ntol::Real=1/train_model.K^2)
	check_corp(corp)
	check_model(train_model)
	(corp.vocab == train_model.corp.vocab)	|| throw(CorpusError("predict corpus and train_model corpus must have identical vocabularies."))
	all([tol, ntol] .>= 0)					|| throw(ArgumentError("tolerance parameters must be nonnegative."))
	all([iter, niter] .>= 0)				|| throw(ArgumentError("iteration parameters must be nonnegative."))

	model = fCTM(corp, train_model.K)
	model.mu = train_model.mu
	model.sigma = train_model.sigma
	model.invsigma = train_model.invsigma
	model.beta = train_model.beta
	model.topics = train_model.topics

	for d in 1:model.M
		for v in 1:iter
			update_phi!(model, d)
			update_tau!(model, d)
			update_logzeta!(model, d)
			update_lambda!(model, d, niter, ntol)
			update_vsq!(model, d, niter, ntol)
			if norm(model.lambda[d] - model.lambda_old[d]) < vtol
				break
			end
		end
	end

	return model
end

function topicdist(model::Union{LDA, gpuLDA, fLDA}, d::Integer)
	(d <= length(model.corp)) || throw(CorpusError("document index outside corpus range."))

	topic_distribution = model.gamma[d] / sum(model.gamma[d])
	return topic_distribution
end

function topicdist(model::Union{CTM, gpuCTM, fCTM}, d::Integer)
	(d <= length(model.corp)) || throw(CorpusError("document index outside corpus range."))

	topic_distribution = additive_logistic(model.lambda[d] + 0.5 * model.vsq[d])
	return topic_distribution
end

function topicdist(model::Union{CTPF, gpuCTPF}, d::Integer)
	(d <= length(model.corp)) || throw(CorpusError("document index outside corpus range."))

	topic_distribution = model.gimel[d] / sum(model.gimel[d])
	return topic_distribution
end

"""
    topicdist(model::TopicModel, doc_indices::Vector{<:Integer})

Get TopicModel topic distributions for document(s) as a probability vector.
"""
function topicdist(model::TopicModel, doc_indices::Vector{<:Integer})
	issubset(doc_indices, 1:length(model.corp)) || throw(CorpusError("some document indices outside corpus range."))

	topic_distributions = Vector{typeof(model.elbo)}[]
	for d in doc_indices
		push!(topic_distributions, topicdist(model, d))
	end

	return topic_distributions
end

topicdist(model::TopicModel, doc_range::UnitRange{<:Integer}) = topicdist(model, collect(doc_range))
