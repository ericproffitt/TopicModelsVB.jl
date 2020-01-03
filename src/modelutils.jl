### Model utilites for TopicModelsVB
### Eric Proffitt
### December 3, 2019

struct TopicModelError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, e::TopicModelError) = print(io, "TopicModelError: ", e.msg)

showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc_indices::Vector{Int}) = showdocs(model.corp, doc_indices)
showdocs(model::TopicModel, doc_range::UnitRange{Int}) = showdocs(model.corp, collect(doc_range))

getvocab(model::TopicModel) = sort(collect(values(model.corp.vocab)))
getusers(model::TopicModel) = sort(collect(values(model.corp.users)))

### Display output for TopicModel objects.
Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::fLDA) = print(io, "Filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::CTM) = print(io, "Correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::fCTM) = print(io, "Filtered correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")
Base.show(io::IO, model::gpuLDA) = print(io, "GPU accelerated latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTM) = print(io, "GPU accelerated correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::gpuCTPF) = print(io, "GPU accelerated collaborative topic Poisson factorization model with $(model.K) topics.")

function check_model(model::LDA)
	"Check LDA parameters."

	check_corp(model.corp) 
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))				|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
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
	"Check fLDA parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))				|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
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
	isequal(size(model.fbeta), (model.K, model.V))									|| throw(TopicModelError("fbeta must be of size (K, V)."))
	(isstochastic(model.fbeta, dims=2) | isempty(model.fbeta))						|| throw(TopicModelError("fbeta must be a right stochastic matrix."))
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
	"Check CTM parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))	
	isequal(model.M, length(model.corp))										|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])					|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])					|| throw(TopicModelError("C must contain sums of document counts."))	
	all(isfinite.(model.mu))													|| throw(TopicModelError("mu must be finite."))
	isequal(size(model.sigma), (model.K, model.K))								|| throw(TopicModelError("sigma must be of size (K, K)."))
	isposdef(model.sigma)														|| throw(TopicModelError("sigma must be positive-definite."))
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
	"Check fCTM parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))										|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])					|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])					|| throw(TopicModelError("C must contain sums of document counts."))
	(0 <= model.eta <= 1)														|| throw(TopicModelError("eta must belong to the interval [0,1]."))
	all(isfinite.(model.mu))													|| throw(TopicModelError("mu must be finite."))
	isequal(size(model.sigma), (model.K, model.K))								|| throw(TopicModelError("sigma must be of size (K, K)."))
	isposdef(model.sigma)														|| throw(TopicModelError("sigma must be positive-definite."))
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
	isequal(size(model.fbeta), (model.K, model.V))								|| throw(TopicModelError("fbeta must be of size (K, V)."))
	(isstochastic(model.fbeta, dims=2) | isempty(model.fbeta))					|| throw(TopicModelError("fbeta must be a right stochastic matrix."))
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
	"Check CTPF parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))			|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
	isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))			|| throw(TopicModelError("Corpus users keys must form unit range of length U."))
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
	isequal(model.alef_temp, fill(a, model.K, model.V))							|| throw(TopicModelError("alef_temp must be a fill a matrix of size (K, V)."))
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
	isequal(model.he_temp, fill(e, model.K, model.U))							|| throw(TopicModelError("he_temp must be a fill e matrix of size (K, U)."))
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
	"Check gpuLDA parameters."

	check_corp(model.corp) 
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))							|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
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
	isequal(length(model.Elogtheta_old), model.M)												|| throw(TopicModelError("Elogtheta_old must be of length M."))
	all(Bool[isequal(length(model.Elogtheta_old[d]), model.K) for d in 1:model.M])				|| throw(TopicModelError("Elogtheta_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta_old[d])) for d in 1:model.M])						|| throw(TopicModelError("Elogtheta_old must be finite."))
	all(Bool[all(model.Elogtheta_old[d] .<= 0) for d in 1:model.M])								|| throw(TopicModelError("Elogtheta_old must be nonpositive."))
	isequal(length(model.gamma), model.M)														|| throw(TopicModelError("gamma must be of length M."))
	all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])						|| throw(TopicModelError("gamma must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])								|| throw(TopicModelError("gamma must be finite."))
	all(Bool[all(model.gamma[d] .> 0) for d in 1:model.M])										|| throw(TopicModelError("gamma must be positive."))
	isequal(length(model.phi), model.M)															|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])			|| throw(TopicModelError("phi must contain matrices of sizes (K, N)."))
	all(Bool[isstochastic(model.phi[d], dims=1) | isempty(model.phi[d]) for d in 1:model.M])	|| throw(TopicModelError("phi must contain left stochastic matrices."))
	isfinite(model.elbo)																		|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::gpuCTM)
	"Check gpuCTM parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))							|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
	isequal(model.M, length(model.corp))														|| throw(TopicModelError("M must equal the number of documents in the corpus."))
	isequal(model.N, [length(doc.terms) for doc in model.corp])									|| throw(TopicModelError("N must contain document lengths."))
	isequal(model.C, [sum(doc.counts) for doc in model.corp])									|| throw(TopicModelError("C must contain sums of document counts."))
	(0 <= model.eta <= 1)																		|| throw(TopicModelError("eta must belong to the interval [0,1]."))
	isequal(length(model.alpha), model.K)														|| throw(TopicModelError("alpha must be of length K."))
	all(isfinite.(model.alpha))																	|| throw(TopicModelError("alpha must be finite."))
	all(model.alpha .> 0)																		|| throw(TopicModelError("alpha must be positive."))
	isequal(length(model.kappa), model.V)														|| throw(TopicModelError("kappa must be of length V"))
	(isprobvec(model.kappa) | isempty(model.kappa))												|| throw(TopicModelError("kappa must be a probability vector."))
	isequal(length(model.kappa_old), model.V)													|| throw(TopicModelError("kappa_old must be of length V."))
	(isprobvec(model.kappa_old) | isempty(model.kappa_old))										|| throw(TopicModelError("kappa_old must be a probability vector."))
	isequal(model.kappa_temp, zeros(model.V))													|| throw(TopicModelError("kappa_temp must be a zero vector of length V."))
	isequal(size(model.beta), (model.K, model.V))												|| throw(TopicModelError("beta must be of size (K, V)."))
	(isstochastic(model.beta, dims=2) | isempty(model.beta))									|| throw(TopicModelError("beta must be a right stochastic matrix."))
	isequal(size(model.beta_old), (model.K, model.V))											|| throw(TopicModelError("beta_old must be of size (K, V)."))
	(isstochastic(model.beta_old, dims=2) | isempty(model.beta_old))							|| throw(TopicModelError("beta_old must be a right stochastic matrix."))
	isequal(model.beta_temp, zeros(model.K, model.V))											|| throw(TopicModelError("beta_temp must be a zero matrix of size (K, V)."))
	isequal(size(model.fbeta), (model.K, model.V))												|| throw(TopicModelError("fbeta must be of size (K, V)."))
	(isstochastic(model.fbeta, dims=2) | isempty(model.fbeta))									|| throw(TopicModelError("fbeta must be a right stochastic matrix."))
	isequal(length(model.Elogtheta), model.M)													|| throw(TopicModelError("Elogtheta must be of length M."))
	all(Bool[isequal(length(model.Elogtheta[d]), model.K) for d in 1:model.M])					|| throw(TopicModelError("Elogtheta must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta[d])) for d in 1:model.M])							|| throw(TopicModelError("Elogtheta must be finite."))
	all(Bool[all(model.Elogtheta[d] .<= 0) for d in 1:model.M])									|| throw(TopicModelError("Elogtheta must be nonpositive."))
	isequal(length(model.Elogtheta_old), model.M)												|| throw(TopicModelError("Elogtheta_old must be of length M."))
	all(Bool[isequal(length(model.Elogtheta_old[d]), model.K) for d in 1:model.M])				|| throw(TopicModelError("Elogtheta_old must contain vectors of length K."))
	all(Bool[all(isfinite.(model.Elogtheta_old[d])) for d in 1:model.M])						|| throw(TopicModelError("Elogtheta_old must be finite."))
	all(Bool[all(model.Elogtheta_old[d] .<= 0) for d in 1:model.M])								|| throw(TopicModelError("Elogtheta_old must be nonpositive."))
	isequal(length(model.gamma), model.M)														|| throw(TopicModelError("gamma must be of length M."))
	all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])						|| throw(TopicModelError("gamma must contain vectors of length K."))
	all(Bool[all(isfinite.(model.gamma[d])) for d in 1:model.M])								|| throw(TopicModelError("gamma must be finite."))
	all(Bool[all(model.gamma[d] .> 0) for d in 1:model.M])										|| throw(TopicModelError("gamma must be positive."))
	isequal(length(model.tau), model.M)															|| throw(TopicModelError("tau must be of length M."))
	all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])						|| throw(TopicModelError("tau must contain vectors of lengths N."))
	all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])									|| throw(TopicModelError("tau must belong to the interval [0,1]."))
	isequal(length(model.phi), model.M)															|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])			|| throw(TopicModelError("phi must contain matrices of sizes (K, N)."))
	all(Bool[isstochastic(model.phi[d], dims=1) | isempty(model.phi[d]) for d in 1:model.M])	|| throw(TopicModelError("phi must contain left stochastic matrices."))
	isfinite(model.elbo)																		|| throw(TopicModelError("elbo must be finite."))
	nothing
end

function check_model(model::gpuCTPF)
	"Check gpuCTPF parameters."

	check_corp(model.corp)
	isequal(collect(1:model.V), sort(collect(keys(model.corp.vocab))))					|| throw(TopicModelError("Corpus vocab keys must form unit range of length V."))
	isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))					|| throw(TopicModelError("Corpus users keys must form unit range of length U."))
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
	all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])	|| throw(TopicModelError(""))
	all(Bool[isstochastic(model.phi[d], dims=1) for d in 1:model.M])					|| throw(TopicModelError(""))
	isequal(length(model.xi), model.M)													|| throw(TopicModelError("phi must be of length M."))
	all(Bool[isequal(size(model.xi[d]), (2model.K, model.R[d])) for d in 1:model.M])	|| throw(TopicModelError(""))
	all(Bool[isstochastic(model.xi[d], dims=1) for d in 1:model.M])						|| throw(TopicModelError(""))
	isfinite(model.elbo)																|| throw(TopicModelError("elbo must be finite"))
	nothing	
end

function update_buffer!(model::gpuLDA)
	"Update gpuLDA model data in GPU RAM."

	terms = vcat([doc.terms for doc in model.corp]...) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp]...)
		
	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	J_partial_sums = zeros(Int, model.V + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
	end

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	model.N_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_partial_sums)
	model.J_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_partial_sums)

	@buffer model.alpha
	model.beta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.Elogtheta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.Elogtheta..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.Elogtheta_dist_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=zeros(Float32, model.M + 64 - model.M % 64))
	#model.gamma_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gamma..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.gamma_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (model.M + 64 - model.M % 64))
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (sum(model.N) + 64 - sum(model.N) % 64))
end

function update_buffer!(model::gpuCTM)
	"Update gpuCTM model data in GPU RAM."

	terms = vcat([doc.terms for doc in model.corp]...) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp]...)

	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	J_partial_sums = zeros(Int, model.V + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
	end

	model.C_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=model.C)
	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	model.N_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_partial_sums)
	model.J_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_partial_sums)

	model.newton_temp_buffer = cl.Buffer(Float32, model.context, :rw, model.K^2 * (model.M + 64 - model.M % 64))
	model.newton_grad_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (model.M + 64 - model.M % 64))
	model.newton_invhess_buffer = cl.Buffer(Float32, model.context, :rw, model.K^2 * (model.M + 64 - model.M % 64))

	@buffer model.sigma
	@buffer model.invsigma
	model.mu_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.mu)
	model.beta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.lambda_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.lambda..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.vsq_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.vsq..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.logzeta_buffer = cl.Buffer(Float32, model.context, :rw, model.M + 64 - model.M % 64)
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (sum(model.N) + 64 - sum(model.N) % 64))
end

function update_buffer!(model::gpuCTPF)
	"Update gpuCTPF model data in GPU RAM."
		
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

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	J_partial_sums = zeros(Int, model.V + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
	end

	R_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		R_partial_sums[d+1] = R_partial_sums[d] + model.R[d]
	end
		
	Y_partial_sums = zeros(Int, model.U + 1)
	for u in 1:model.U
		Y_partial_sums[u+1] = Y_partial_sums[u] + Y[u]
	end

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms)
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	model.readers_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=readers)
	model.ratings_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=ratings)
	model.readers_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=readers_sortperm)

	model.N_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_partial_sums)
	model.J_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_partial_sums)
	model.R_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=R_partial_sums)
	model.Y_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=Y_partial_sums)

	model.alef_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alef)
	model.he_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.he)
	model.bet_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.bet)
	model.vav_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.vav)
	model.gimel_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gimel..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.zayin_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.zayin..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.dalet_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.dalet)
	model.het_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.het)
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (sum(model.N) + 64 - sum(model.N) % 64))
	model.xi_buffer = cl.Buffer(Float32, model.context, :rw, 2 * model.K * (sum(model.R) + 64 - sum(model.R) % 64))
end

function update_host!(model::TopicModel)
	nothing
end

function update_host!(model::gpuLDA)
	"Update gpuLDA model data in CPU RAM."

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	model.beta = reshape(cl.read(model.queue, model.beta_buffer), model.K, model.V)
	
	@host model.Elogtheta_buffer

	gamma_host = reshape(cl.read(model.queue, model.gamma_buffer), model.K, model.M + 64 - model.M % 64)
	model.gamma = [gamma_host[:,d] for d in 1:model.M]

	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N) + 64 - sum(model.N) % 64)
	model.phi = [phi_host[:,N_partial_sums[d]+1:N_partial_sums[d+1]] for d in 1:model.M]
end

function update_host!(model::gpuCTM)
	"Update gpuCTM model data in CPU RAM."

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	@host model.mu_buffer
	model.sigma = reshape(cl.read(model.queue, model.sigma_buffer), model.K, model.K)
	model.invsigma = reshape(cl.read(model.queue, model.invsigma_buffer), model.K, model.K)
	model.beta = reshape(cl.read(model.queue, model.beta_buffer), model.K, model.V)
	@host model.lambda_buffer
	@host model.vsq_buffer
	model.logzeta = cl.read(model.queue, model.logzeta_buffer)

	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N) + 64 - sum(model.N) % 64)
	model.phi = [phi_host[:,N_partial_sums[d]+1:N_partial_sums[d+1]] for d in 1:model.M]
end

function update_host!(model::gpuCTPF)
	"Update gpuCTPF model data in CPU RAM."

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	R_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		R_partial_sums[d+1] = R_partial_sums[d] + model.R[d]
	end

	model.alef = reshape(cl.read(model.queue, model.alef_buffer), model.K, model.V)
	model.he = reshape(cl.read(model.queue, model.he_buffer), model.K, model.U)
	model.bet = cl.read(model.queue, model.bet_buffer)
	model.vav = cl.read(model.queue, model.vav_buffer)
	@host model.gimel_buffer
	
	zayin_host = reshape(cl.read(model.queue, model.zayin_buffer), model.K, model.M + 64 - model.M % 64)
	model.zayin = [zayin_host[:,d] for d in 1:model.M]
		
	model.dalet = cl.read(model.queue, model.dalet_buffer)
	model.het = cl.read(model.queue, model.het_buffer)
	
	phi_host = reshape(cl.read(model.queue, model.phi_buffer), model.K, sum(model.N) + 64 - sum(model.N) % 64)
	model.phi = [phi_host[:,N_partial_sums[d]+1:N_partial_sums[d+1]] for d in 1:model.M]

	xi_host = reshape(cl.read(model.queue, model.xi_buffer), 2 * model.K, sum(model.R) + 64 - sum(model.R) % 64)
	model.xi = [xi_host[:,R_partial_sums[d]+1:R_partial_sums[d+1]] for d in 1:model.M]
end

function check_elbo!(model::TopicModel, check_elbo::Real, k::Int, tol::Real)
	"Check and print value of delta_elbo."
	"If abs(delta_elbo) < tol, terminate algorithm."

	if k % check_elbo == 0
		update_host!(model)
		delta_elbo = -(model.elbo - update_elbo!(model))
		println(k, " ∆elbo: ", round(delta_elbo, digits=3))

		if abs(delta_elbo) < tol
			return true
		end
	end
	false
end

function gendoc(model::Union{LDA, gpuLDA}, laplace_smooth::Real=0.0)
	"Generate artificial document from LDA or gpuLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((model.beta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms=terms, counts=counts)
end

function gendoc(model::fLDA, laplace_smooth::Real=0.0)
	"Generate artificial document from fLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((model.fbeta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms=terms, counts=counts)
end

function gendoc(model::Union{CTM, gpuCTM}, laplace_smooth::Real=0.0)
	"Generate artificial document from CTM or gpuCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp.(theta) / sum(exp.(theta))
	topicdist = Categorical(theta)
	lexdist = [Categorical((model.beta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms=terms, counts=counts)
end

function gendoc(model::fCTM, laplace_smooth::Real=0.0)
	"Generate artificial document from fCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp(theta) / sum(exp(theta))
	topicdist = Categorical(theta)
	lexdist = [Categorical((model.fbeta[i,:] .+ laplace_smooth) / (1 + laplace_smooth * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms=terms, counts=counts)
end

function gencorp(model::TopicModel, corp_size::Integer; laplace_smooth::Real=0.0)
	"Generate artificial corpus using specified generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	corp_size > 0 		|| throw(ArgumentError("corp_size parameter must be a positive integer."))
	laplace_smooth >= 0	|| throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	corp = Corpus(vocab=model.corp.vocab, users=model.corp.users)
	corp.docs = [gendoc(model, laplace_smooth) for d in 1:corp_size]
	return corp
end

function showtopics(model::TopicModel, top_n_terms::Integer=min(15, model.V); topics::Union{<:Integer, Vector{<:Integer}}=collect(1:model.K), cols::Integer=4)
	"Display the top n terms for each topic."
	"topics parameter controls which topics are displayed."
	"cols parameter controls the number of topic columns displayed per line."

	top_n_terms <= model.V					|| throw(ArgumentError("Number of displayed terms must be less than vocab size."))
	checkbounds(Bool, 1:model.K, topics)	|| throw(ArgumentError("Some topic indices are outside range."))
	cols > 0								|| throw(ArgumentError("cols must be a positive integer."))
	
	cols = min(cols, length(topics))

	vocab = model.corp.vocab
	maxjspacings = [maximum([length(vocab[j]) for j in topic[1:top_n_terms]]) for topic in model.topics]

	for block in Iterators.partition(topics, cols)
		for j in 0:top_n_terms
			for (k, i) in enumerate(block)
				if j == 0
					jspacing = max(4, maxjspacings[i] - length("$i") - 2)
					k == cols ? print(Crayon(foreground=:yellow, bold=true), "topic $i") : print(Crayon(foreground=:yellow, bold=true), "topic $i" * " "^jspacing)
				else
					jspacing = max(6 + length("$i"), maxjspacings[i]) - length(vocab[model.topics[i][j]]) + 4
					k == cols ? print(Crayon(foreground=:white, bold=false), vocab[model.topics[i][j]]) : print(Crayon(foreground=:white, bold=false), vocab[model.topics[i][j]] * " "^jspacing)
				end
			end
			println()
		end
		println()
	end
end

function showlibs(model::Union{CTPF, gpuCTPF}, users::Vector{<:Integer})
	"Display the documents in a user(s) library."

	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("Some user indices are outside range."))
	
	for u in users
		@juliadots "user $u\n"
		try
			if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		
		catch
			@juliadots model.corp.users[u] * "\n"
		end
		
		for d in model.libs[u]
			print(Crayon(foreground=:yellow, bold=true), " • ")
			isempty(model.corp[d].title) ? print(Crayon(foreground=:white, bold=true), "doc $d\n") : print(Crayon(foreground=:white, bold=false), "$(model.corp[d].title)\n")
		end
		print()
	end
end

showlibs(model::Union{CTPF, gpuCTPF}, user::Integer) = showlibs(model, [user])

function showdrecs(model::Union{CTPF, gpuCTPF}, docs::Union{Integer, Vector{<:Integer}}, U::Integer=min(16, model.U); cols::Integer=4)
	"Display the top U user recommendations for a document(s)."
	"cols parameter controls the number of topic columns displayed per line."

	checkbounds(Bool, 1:model.U, U) 	|| throw(ArgumentError("Some user indices are outside range."))
	checkbounds(Bool, 1:model.M, docs) 	|| throw(ArgumentError("Some document indices are outside range."))
	cols > 0							|| throw(ArgumentError("cols must be a positive integer."))
	isa(docs, Vector) || (docs = [docs])
	corp, drecs, users = model.corp, model.drecs, model.corp.users

	for d in docs
		@juliadots "doc $d\n"
		if !isempty(corp[d].title)
			@juliadots corp[d].title * "\n"
		end

		usercols = collect(Iterators.partition(drecs[d][1:U], Int(ceil(U / cols))))
		rankcols = collect(Iterators.partition(1:U, Int(ceil(U / cols))))

		for i in 1:length(usercols[1])
			for j in 1:length(usercols)
				try
					uspacing = maximum([length(users[u]) for u in usercols[j]]) - length(users[usercols[j][i]]) + 4
					rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
					print(Crayon(foreground=:yellow, bold=true), string(rankcols[j][i]) * ". " * " "^rspacing)
					j == length(usercols) ? print(Crayon(foreground=:white, bold=false), users[usercols[j][i]]) : print(Crayon(foreground=:white, bold=false), users[usercols[j][i]] * " "^uspacing)
				
				catch
					nothing
				end
			end
			println()
		end
		println()
	end
end

function showurecs(model::Union{CTPF, gpuCTPF}, users::Union{Integer, Vector{<:Integer}}, M::Integer=min(10, model.M); cols::Integer=1)
	"# Show the top 'M' document recommendations for a user(s)."
	"If a document has no title, the document's index in the corpus will be shown instead."

	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("Some user indices are outside range."))
	checkbounds(Bool, 1:model.M, M) 	|| throw(ArgumentError("Some document indices are outside range."))
	cols > 0
	isa(users, Vector) || (users = [users])

	corp, urecs, docs = model.corp, model.urecs, model.corp.docs

	for u in users
		@juliadots "user $u\n"
		try 
			if corp.users[u][1:5] != "#user"
				@juliadots corp.users[u] * "\n"
			end
		
		catch 
			@juliadots corp.users[u] * "\n"
		end

		docucols = collect(Iterators.partition(urecs[u][1:M], Int(ceil(M / cols))))
		rankcols = collect(Iterators.partition(1:M, Int(ceil(M / cols))))

		for i in 1:length(docucols[1])
			for j in 1:length(docucols)
				try
					!isempty(corp[docucols[j][i]].title) ? title = corp[docucols[j][i]].title : title = "doc $(docucols[j][i])"
					dspacing = maximum([max(4 + length("$(docucols[j][i])"), length(docs[d].title)) for d in docucols[j]]) - length(title) + 4
					rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
					print(Crayon(foreground=:yellow, bold=true), string(rankcols[j][i]) * ". " * " "^rspacing)
					j == length(docucols) ? print(Crayon(foreground=:white, bold=false), title) : print(Crayon(foreground=:white, bold=false), title * " "^dspacing)

				catch
					nothing
				end
			end
			println()
		end
		println()
	end
end

function predict(corp::Corpus, model_train::TopicModel)
	nothing
end







