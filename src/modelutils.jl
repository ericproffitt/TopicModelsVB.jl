showdocs(model::TopicModel, ds::Vector{Int}) = showdocs(model.corp, ds)
showdocs(model::TopicModel, docs::Vector{Document}) = showdocs(model.corp, docs)
showdocs(model::TopicModel, ds::UnitRange{Int}) = showdocs(model.corp, collect(ds))
showdocs(model::TopicModel, d::Int) = showdocs(model.corp, d)
showdocs(model::TopicModel, doc::Document) = showdocs(model.corp, doc)

getlex(model::TopicModel) = sort(collect(values(model.corp.lex)))
getusers(model::TopicModel) = sort(collect(values(model.corp.users)))

Base.show(io::IO, model::LDA) = print(io, "Latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::fLDA) = print(io, "Filtered latent Dirichlet allocation model with $(model.K) topics.")
Base.show(io::IO, model::CTM) = print(io, "Correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::fCTM) = print(io, "Filtered correlated topic model with $(model.K) topics.")
Base.show(io::IO, model::DTM) = print(io, "Dynamic topic model with $(model.K) topics and ∆ = $(model.delta).")
Base.show(io::IO, model::jDTM) = print(io, "Dynamic topic model with $(model.K) topics and ∆ = $(model.delta).")
Base.show(io::IO, model::CTPF) = print(io, "Collaborative topic Poisson factorization model with $(model.K) topics.")

function checkmodel(model::LDA)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.beta, 1), model.K)
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert isfinite(model.elbo)
end

function checkmodel(model::fLDA)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.alpha))
	@assert all(ispositive(model.alpha))
	@assert isequal(length(model.alpha), model.K)
	@assert (0 <= model.eta <= 1)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert isfinite(model.elbo)
end

function checkmodel(model::CTM)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert all(isfinite(model.mu))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])
	@assert all(isfinite(model.lzeta))
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert isfinite(model.elbo)
end

function checkmodel(model::fCTM)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert (0 <= model.eta <= 1)
	@assert all(isfinite(model.mu))
	@assert isposdef(model.sigma)
	@assert isequal(size(model.beta), (model.K, model.V))
	@assert isprobvec(model.beta, 2)
	@assert isequal(size(model.fbeta), (model.K, model.V))
	@assert isprobvec(model.fbeta, 2)
	@assert isequal(length(model.kappa), model.V)
	@assert isprobvec(model.kappa)
	@assert all(Bool[all(isfinite(model.lambda[d])) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.vsq[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.vsq[d])) for d in 1:model.M])
	@assert all(isfinite(model.lzeta))
	@assert all(Bool[isequal(length(model.tau[d]), model.N[d]) for d in 1:model.M])
	@assert all(Bool[all(0 .<= model.tau[d] .<= 1) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert isfinite(model.elbo)
end

function checkmodel(model::DTM)	
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert !isnegative(model.T)
	@assert isequal(vcat(model.S...), sortperm([doc.stamp for doc in model.corp]))
	@assert isfinite(model.sigmasq)
	@assert ispositive(model.sigmasq)
	@assert all(Bool[all(isfinite(model.alpha[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.alpha[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(isfinite(model.m0))
	@assert all(isfinite(model.v0))
	@assert all(ispositive(model.v0))
	@assert all(Bool[all(isfinite(model.m[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.v[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.v[t])) for t in 1:model.T])
	@assert all(isfinite(model.bsq))
	@assert all(ispositive(model.bsq))
	@assert all(Bool[all(isfinite(model.betahat[t])) for t in 1:model.T])
	@assert all(isfinite(model.mbeta0))
	@assert all(isfinite(model.vbeta0))
	@assert all(ispositive(model.vbeta0))
	@assert all(Bool[all(isfinite(model.mbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.vbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.vbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.Eexpbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.Eexpbeta[t])) for t in 1:model.T])
	@assert all(isfinite(model.a))
	@assert all(Bool[all(isfinite(model.rEexpbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.rEexpbeta[t])) for t in 1:model.T])
	@assert all(isfinite(model.lzeta))
	@assert isfinite(model.delta)
	@assert ispositive(model.delta)
	@assert isfinite(model.elbo)
end

function checkmodel(model::jDTM)	
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(doc.terms) for doc in model.corp])
	@assert isequal(model.C, [sum(doc.counts) for doc in model.corp])
	@assert !isnegative(model.T)
	@assert isequal(vcat(model.S...), sortperm([doc.stamp for doc in model.corp]))
	@assert isfinite(model.sigmasq)
	@assert ispositive(model.sigmasq)
	@assert all(Bool[all(isfinite(model.alpha[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.alpha[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gamma[d])) for d in 1:model.M])
	@assert all(Bool[isequal(length(model.gamma[d]), model.K) for d in 1:model.M])
	@assert all(Bool[isequal(size(model.phi[d]), (model.K, model.N[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(isfinite(model.m0))
	@assert all(isfinite(model.v0))
	@assert all(ispositive(model.v0))
	@assert all(Bool[all(isfinite(model.m[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.v[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.v[t])) for t in 1:model.T])
	@assert all(isfinite(model.bsq))
	@assert all(ispositive(model.bsq))
	@assert all(Bool[all(isfinite(model.betahat[t])) for t in 1:model.T])
	@assert all(isfinite(model.mbeta0))
	@assert all(isfinite(model.vbeta0))
	@assert all(ispositive(model.vbeta0))
	@assert all(Bool[all(isfinite(model.mbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.vbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.vbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(isfinite(model.Eexpbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.Eexpbeta[t])) for t in 1:model.T])
	@assert all(isfinite(model.a))
	@assert all(Bool[all(isfinite(model.rEexpbeta[t])) for t in 1:model.T])
	@assert all(Bool[all(ispositive(model.rEexpbeta[t])) for t in 1:model.T])
	@assert isfinite(model.delta)
	@assert ispositive(model.delta)
	@assert isfinite(model.elbo)
end

function checkmodel(model::CTPF)
	checkcorp(model.corp)
	@assert isequal(collect(1:model.V), sort(collect(keys(model.corp.lex))))	
	@assert isequal(collect(1:model.U), sort(collect(keys(model.corp.users))))
	@assert isequal(model.M, length(model.corp))
	@assert isequal(model.N, [length(model.corp[d].terms) for d in 1:model.M])
	@assert isequal(model.C, [sum(model.corp[d].counts) for d in 1:model.M])
	@assert isequal(model.R, [length(model.corp[d].readers) for d in 1:model.M])
	@assert ispositive(model.a)
	@assert ispositive(model.b)
	@assert ispositive(model.c)
	@assert ispositive(model.d)
	@assert ispositive(model.e)
	@assert ispositive(model.f)
	@assert ispositive(model.g)
	@assert ispositive(model.h)
	@assert all(isfinite(model.alef))
	@assert all(ispositive(model.alef))
	@assert all(isfinite(model.bet))
	@assert all(ispositive(model.bet))
	@assert all(Bool[all(isfinite(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.gimel[d])) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.dalet[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.dalet[d])) for d in 1:model.M])
	@assert all(isfinite(model.he))
	@assert all(ispositive(model.he))
	@assert all(isfinite(model.vav))
	@assert all(ispositive(model.vav))
	@assert all(Bool[all(isfinite(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.zayin[d])) for d in 1:model.M])
	@assert all(Bool[all(isfinite(model.het[d])) for d in 1:model.M])
	@assert all(Bool[all(ispositive(model.het[d])) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.phi[d], 1) for d in 1:model.M])
	@assert all(Bool[isprobvec(model.xi[d], 1) for d in 1:model.M])
	@assert isfinite(model.elbo)
end

function checkELBO!(model::TopicModel, k::Int, chkelbo::Int, tol::Float64)
	converged = false
	if k % chkelbo == 0
		∆elbo = -(model.elbo - updateELBO!(model))
		println(k, " ∆elbo: ", round(∆elbo, 3))
		if abs(∆elbo) < tol
			converged = true
		end
	end

	return converged
end

function gendoc(model::LDA, a::Real=0.0)
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

function gendoc(model::fLDA, a::Real=0.0)
	@assert !isnegative(a)
	
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

function gendoc(model::CTM, a::Real=0.0)
	@assert !isnegative(a)
	
	C = rand(Poisson(mean(model.C)))
	termcount = Dict{Int, Int}()
	theta = rand(MvNormal(model.mu, model.sigma))
	theta = exp(theta) / sum(exp(theta))
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

function gendoc(model::fCTM, a::Real=0.0)
	@assert !isnegative(a)
	
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

function gencorp(model::Union{LDA, fLDA, CTM, fCTM}, corpsize::Int, a::Real=0.0)
	@assert ispositive(corpsize)
	@assert !isnegative(a)
	
	corp = Corpus(lex=model.corp.lex, users=model.corp.users)
	corp.docs = [gendoc(model, a) for d in 1:corpsize]

	return corp
end

function showtopics(model::TopicModel, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), cols::Int=4)
	@assert checkbounds(Bool, model.V, N)
	@assert checkbounds(Bool, model.K, topics)
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

function showtopics(model::Union{DTM, jDTM}, N::Int=min(15, model.V); topics::Union{Int, Vector{Int}}=collect(1:model.K), times::Union{Int, Vector{Int}}=collect(1:model.T), cols::Int=4)
	@assert checkbounds(Bool, model.V, N)
	@assert checkbounds(Bool, model.K, topics)
	@assert checkbounds(Bool, model.T, times)
	@assert ispositive(cols)
	isa(times, Vector) || (times = [times])
	
	corp, lex = model.corp, model.corp.lex

	if length(topics) > 1
		container = LDA(Corpus(), model.K)
		for t in times
			container.corp = corp
			container.topics = model.topics[t][topics]
			container.V = model.V
			@juliadots "Time: $t\n"
			@juliadots "Span: $(corp[model.S[t][1]].stamp) - $(corp[model.S[t][end]].stamp)\n"
			showtopics(container, N, topics=topics, cols=cols)
		end
	
	else
		cols = min(cols, length(times))
		@juliadots "Topic: $(topics[1])\n"
		maxjspacings = [maximum([length(lex[j]) for j in time[topics[1]][1:N]]) for time in model.topics]

		for block in partition(times, cols)
			for j in 0:N
				for (s, t) in enumerate(block)
					if j == 0
						jspacing = max(4, maxjspacings[t] - length("$t") - 1)
						s == cols ? yellow("time $t") : yellow("time $t" * " "^jspacing)
					else
						jspacing = max(5 + length("$t"), maxjspacings[t]) - length(lex[model.topics[t][topics[1]][j]]) + 4
						s == cols ? print(lex[model.topics[t][topics[1]][j]]) : print(lex[model.topics[t][topics[1]][j]] * " "^jspacing)
					end
				end
				println()
			end
			println()
		end
	end
end

function showlibs(model::CTPF, users::Vector{Int})
	@assert checkbounds(Bool, model.U, users)
	
	for u in users
		@juliadots "User: $u\n"
		try if model.corp.users[u][1:5] != "#user"
				@juliadots model.corp.users[u] * "\n"
			end
		catch @juliadots model.corp.users[u] * "\n"
		end
		
		for d in model.libs[u]
			yellow(" • ")
			isempty(model.corp[d].title) ? bold("Doc: $d\n") : bold("$(model.corp[d].title)\n")
		end
		println()
	end
end

showlibs(model::CTPF, user::Int) = showlibs(model, [user])

function showdrecs(model::CTPF, docs::Union{Int, Vector{Int}}, U::Int=min(16, model.U); cols::Int=4)
	@assert checkbounds(Bool, model.M, docs)	
	@assert checkbounds(Bool, model.U, U)
	@assert ispositive(cols)
	isa(docs, Vector) || (docs = [docs])
	corp, drecs, users = model.corp, model.drecs, model.corp.users

	for d in docs
		@juliadots "Doc: $d\n"
		if !isempty(corp[d].title)
			@juliadots corp[d].title * "\n"
		end

		usercols = partition(drecs[d][1:U], Int(ceil(U / cols)))
		rankcols = partition(1:U, Int(ceil(U / cols)))

		for i in 1:length(usercols[1])
			for j in 1:length(usercols)
				try
				uspacing = maximum([length(users[u]) for u in usercols[j]]) - length(users[usercols[j][i]]) + 4
				rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
				yellow(string(rankcols[j][i]) * ". " * " "^rspacing)
				j == length(usercols) ? print(users[usercols[j][i]]) : print(users[usercols[j][i]] * " "^uspacing)
				end
			end
			println()
		end
		println()
	end
end

function showurecs(model::CTPF, users::Union{Int, Vector{Int}}, M::Int=min(10, model.M); cols::Int=1)
	@assert checkbounds(Bool, model.U, users)
	@assert checkbounds(Bool, model.M, M)
	@assert ispositive(cols)
	checkmodel(model)
	isa(users, Vector) || (users = [users])
	corp, urecs, docs = model.corp, model.urecs, model.corp.docs

	for u in users
		@juliadots "User: $u\n"
		try if corp.users[u][1:5] != "#user"
				@juliadots corp.users[u] * "\n"
			end
		catch @juliadots corp.users[u] * "\n"
		end

		docucols = partition(urecs[u][1:M], Int(ceil(M / cols)))
		rankcols = partition(1:M, Int(ceil(M / cols)))

		for i in 1:length(docucols[1])
			for j in 1:length(docucols)
				try
				!isempty(corp[docucols[j][i]].title) ? title = corp[docucols[j][i]].title : title = "doc $(docucols[j][i])"
				dspacing = maximum([max(4 + length("$(docucols[j][i])"), length(docs[d].title)) for d in docucols[j]]) - length(title) + 4
				rspacing = maximum([length("$r") for r in rankcols[j]]) - length(string(rankcols[j][i]))
				yellow(string(rankcols[j][i]) * ". " * " "^rspacing)
				j == length(docucols) ? bold(title) : bold(title * " "^dspacing)
				end
			end
			println()
		end
		println()
	end
end

