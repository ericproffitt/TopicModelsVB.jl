### Model utilites for TopicModelsVB
### Eric Proffitt
### December 3, 2019

showdocs(model::TopicModel, doc_indices::Vector{Int}) = showdocs(model.corp, doc_indices)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, doc_range::UnitRange{Int}) = showdocs(model.corp, collect(doc_range))
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)

getlex(model::TopicModel) = sort(collect(values(model.corp.vocab)))
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

function update_buffer!(model::gpuLDA)
	"Update gpuLDA model data in GPU RAM."

	terms = vcat([doc.terms for doc in model.corp) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp)
		
	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	J_partial_sums = zeros(Int, model.M + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
	end

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms))
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	N_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_partial_sums)
	J_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_partial_sums)

	@buffer model.alpha
	model.beta_buffer = cl.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
	model.Elogtheta_buffer = cl.Buffer(Float32, model.context, :rw, hcat(model.Elogtheta..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.gamma_buffer = cl.Buffer(Float32, model.context, :rw, zeros(Float32, model.K * model.M + 64 - model.M % 64))
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, zeros(Float32, model.K, sum(model.N) + 64 - sum(model.N) % 64))
end

function update_buffer!(model::gpuCTM)
	"Update gpuCTM model data in GPU RAM."

	terms = vcat([doc.terms for doc in model.corp) .- 1
	terms_sortperm = sortperm(terms) .- 1
	counts = vcat([doc.counts for doc in model.corp)

	J = zeros(Int, model.V)
	for j in terms
		J[j+1] += 1
	end

	N_partial_sums = zeros(Int, model.M + 1)
	for d in 1:model.M
		N_partial_sums[d+1] = N_partial_sums[d] + model.N[d]
	end

	J_partial_sums = zeros(Int, model.M + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
	end

	model.terms_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms))
	model.terms_sortperm_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=terms_sortperm)
	model.counts_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=counts)

	N_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=N_partial_sums)
	J_partial_sums_buffer = cl.Buffer(Int, model.context, (:r, :copy), hostbuf=J_partial_sums)

	model.newton_temp_buffer = cl.Buffer(Float32, model.context, :rw, model.K^2 * (model.M + 64 - model.M % 64)))
	model.newton_grad_buffer = cl.Buffer(Float32, model.context, :rw, model.K * (model.M + 64 - model.M % 64)))
	model.newton_invhess_buffer = cl.Buffer(Float32, model.context, :rw, model.K^2 * (model.M + 64 - model.M % 64)))

	@buffer model.sigma
	@buffer model.invsigma
	model.mu_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).mu))
	model.lambda_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat(model.lambda..., zeros(Float32, model.K, 64 - model.M % 64))))
	model.vsq_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat(model.vsq..., zeros(Float32, model.K, 64 - model.M % 64)))
	model.logzeta_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=model.logzeta))
	model.phi_buffer = cl.Buffer(Float32, model.context, :rw, zeros(Float32, model.K, sum(model.N) + 64 - sum(model.N) % 64))
end

function update_buffer!(model::gpuCTPF)
	"Update gpuCTPF model data in GPU RAM."

	elseif expr.args[2] == :(:readers)
		expr_out = :($(esc(model)).readers_buffer = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).readers))

	elseif expr.args[2] == :(:ratings)
		expr_out = :($(esc(model)).ratings_buffer = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).ratings))

	elseif expr.args[2] == :(:views)
		expr_out = :($(esc(model)).views_buffer = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).views))

	@buffer model.Npsums
	@buffer model.Jpsums
	@buffer model.Rpsums
	@buffer model.Ypsums
	@buffer model.terms
	@buffer model.counts
	@buffer model.words
	@buffer model.readers
	@buffer model.ratings
	@buffer model.views
	@buffer model.phi
	@buffer model.xi
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

	J_partial_sums = zeros(Int, model.M + 1)
	for j in 1:model.V
		J_partial_sums[j+1] = J_partial_sums[j] + J[j]
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

	@host model.mubuf
	@host model.sigmabuf
	@host model.invsigmabuf
	@host model.betabuf
	@host model.lambdabuf
	@host model.vsqbuf
	@host model.lzetabuf
	@host model.phibuf
end

function update_host!(model::gpuCTPF)
	"Update gpuCTPF model data in CPU RAM."

	@host model.alefbuf
	@host model.betbuf
	@host model.gimelbuf
	@host model.daletbuf
	@host model.hebuf
	@host model.vavbuf
	@host model.zayinbuf
	@host model.hetbuf
	@host model.phibuf
	@host model.xibuf
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

function gendoc(model::AbstractLDA, laplace_smooth::Real=0.0)
	"Generate artificial document from LDA or gpuLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
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

function gendoc(model::AbstractfLDA, laplace_smooth::Real=0.0)
	"Generate artificial document from fLDA generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(Dirichlet(model.alpha))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.fbeta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gendoc(model::AbstractCTM, laplace_smooth::Real=0.0)
	"Generate artificial document from CTM or gpuCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp.(theta) / sum(exp.(theta))
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

function gendoc(model::AbstractfCTM, laplace_smooth::Real=0.0)
	"Generate artificial document from fCTM generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp(theta) / sum(exp(theta))
	topicdist = Categorical(theta)
	lexdist = [Categorical((vec(model.fbeta[i,:]) + a) / (1 + a * model.V)) for i in 1:model.K]
	for _ in 1:C
		z = rand(topicdist)
		w = rand(lexdist[z])
		haskey(termcount, w) ? termcount[w] += 1 : termcount[w] = 1
	end
	terms = collect(keys(termcount))
	counts = collect(values(termcount))

	return Document(terms, counts=counts)
end

function gencorp(model::Union{AbstractLDA, AbstractfLDA, AbstractCTM, AbstractfCTM}, corp_size::Integer, laplace_smooth::Real=0.0)
	"Generate artificial corpus using specified generative model."
	"laplace_smooth governs the amount of Laplace smoothing applied to the topic-term distribution."

	corp_size > 0 || throw(ArgumentError("corp_size parameter must be a positive integer."))
	laplace_smooth >= 0 || throw(ArgumentError("laplace_smooth parameter must be nonnegative."))
	
	corp = Corpus(vocab=model.corp.vocab, users=model.corp.users)
	corp.docs = [gendoc(model, laplace_smooth) for d in 1:corp_size]
	return corp
end

function showtopics(model::TopicModel, N::Integer=min(15, model.V); topics::Union{T, Vector{T}}=collect(1:model.K), cols::Integer=4)
	"Display the top N terms for each topic."
	"topics parameter controls which topics are displayed."
	"cols parameter controls the number of topic columns displayed per line."

	checkbounds(Bool, 1:model.V, N) || throw(ArgumentError("Some vocab indices are outside range."))
	checkbounds(Bool, 1:model.K, topics) || throw(ArgumentError("Some topic indices are outside range."))
	cols > 0 || throw(ArgumentError("cols must be a positive integer."))
	cols = min(cols, length(topics))

	vocab = model.corp.vocab
	maxjspacings = [maximum([length(vocab[j]) for j in topic[1:N]]) for topic in model.topics]

	for block in partition(topics, cols)
		for j in 0:N
			for (k, i) in enumerate(block)
				if j == 0
					jspacing = max(4, maxjspacings[i] - length("$i") - 2)
					k == cols ? yellow("topic $i") : yellow("topic $i" * " "^jspacing)
				else
					jspacing = max(6 + length("$i"), maxjspacings[i]) - length(vocab[model.topics[i][j]]) + 4
					k == cols ? print(vocab[model.topics[i][j]]) : print(vocab[model.topics[i][j]] * " "^jspacing)
				end
			end
			println()
		end
		println()
	end
end

showtopics(model::TopicModel, N::Integer; topic::Integer, cols::Integer) = showtopics(model, N, topics=[topic], cols=cols)

function showlibs(model::CTPF, users::Vector{<:Integer})
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

showlibs(model::CTPF, user::Integer) = showlibs(model, [user])

function showdrecs(model::CTPF, docs::Union{Integer, Vector{<:Integer}}, U::Integer=min(16, model.U); cols::Integer=4)
	"Display the top U user recommendations for a document(s)."
	"cols parameter controls the number of topic columns displayed per line."

	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("Some user indices are outside range."))
	checkbounds(Bool, 1:model.M, docs) || throw(ArgumentError("Some document indices are outside range."))
	@assert cols > 0
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

function showurecs(model::CTPF, users::Union{Integer, Vector{<:Integer}}, M::Integer=min(10, model.M); cols::Integer=1)
	"# Show the top 'M' document recommendations for a user(s)."
	"If a document has no title, the document's index in the corpus will be shown instead."

	checkbounds(Bool, 1:model.U, users) || throw(ArgumentError("Some user indices are outside range."))
	checkbounds(Bool, 1:model.M, M) || throw(ArgumentError("Some document indices are outside range."))
	@assert cols > 0
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
					yellow(string(rankcols[j][i]) * ". " * " "^rspacing)
					j == length(docucols) ? bold(title) : bold(title * " "^dspacing)

				catch
					nothing
				end
			end
			println()
		end
		println()
	end
end

