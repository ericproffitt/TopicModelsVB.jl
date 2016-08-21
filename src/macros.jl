macro juliadots(expr::Expr)
	expr = :(print_with_color(:red, " ●");
				print_with_color(:green, "●");
				print_with_color(:blue, "● ");
				print_with_color(:bold, $expr))
	return expr
end

macro boink(expr::Expr)
	expr = :($expr + EPSILON)
	return expr
end

macro bumper(expr::Expr)
	if expr.head == :.
		expr = :($expr += EPSILON)
	elseif expr.head == :(=)
		expr = :($(expr.args[1]) = EPSILON + $(expr.args[2]))
	end
	return expr
end

macro buf(args...)
	if isa(args[1], Expr)
		expr = args[1]
	else
		b = args[1]
		expr = args[2]
	end

	if expr.args[2] == :(:Npsums)
		quoteblock =
		quote
		model.Npsumsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.Npsums[b])
		end

	elseif expr.args[2] == :(:Jpsums)
		quoteblock =
		quote
		model.Jpsumsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.Jpsums[b])
		end

	elseif expr.args[2] == :(:Rpsums)
		quoteblock =
		quote
		model.Rpsumsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.Rpsums[b])
		end

	elseif expr.args[2] == :(:Ypsums)
		quoteblock =
		quote
		model.Ypsumsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.Ypsums[b])
		end

	elseif expr.args[2] == :(:terms)
		quoteblock =
		quote
		model.termsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.terms[b])
		end

	elseif expr.args[2] == :(:counts)
		quoteblock =
		quote
		model.countsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.counts[b])
		end

	elseif expr.args[2] == :(:words)
		quoteblock =
		quote
		model.wordsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.words[b])
		end

	elseif expr.args[2] == :(:readers)
		quoteblock =
		quote
		model.readersbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.readers[b])
		end

	elseif expr.args[2] == :(:ratings)
		quoteblock =
		quote
		model.ratingsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.ratings[b])
		end

	elseif expr.args[2] == :(:views)
		quoteblock =
		quote
		model.viewsbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.views[b])
		end

	elseif expr.args[2] == :(:alpha)
		quoteblock =
		quote
		model.alphabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alpha)
		end

	elseif expr.args[2] == :(:beta)
		quoteblock =
		quote
		model.betabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
		end

	elseif expr.args[2] == :(:newbeta)
		quoteblock =
		quote
		model.newbetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=zeros(Float32, model.K, model.V))
		end

	elseif expr.args[2] == :(:gamma)
		quoteblock =
		quote
		model.gammabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gamma..., zeros(Float32, model.K, 64 - model.M % 64)))
		end

	elseif expr.args[2] == :(:phi)
		quoteblock =
		quote
		batch = model.batches[b]
		model.phibuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=ones(Float32, model.K, sum(model.N[batch]) + 64 - sum(model.N[batch]) % 64) / model.K)
		end

	elseif expr.args[2] == :(:Elogtheta)
		quoteblock = 
		quote
		batch = model.batches[b]
		model.Elogthetabuf = OpenCL.Buffer(Float32, model.context, :rw, model.K * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:Elogthetasum)
		quoteblock =
		quote
		model.Elogthetasumbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.Elogthetasum)
		end

	elseif expr.args[2] == :(:C)
		quoteblock =
		quote
		batch = model.batches[b]
		model.Cbuf = OpenCL.Buffer(Int, model.context, (:r, :copy), hostbuf=model.C[batch])
		end

	elseif expr.args[2] == :(:newtontemp)
		quoteblock =
		quote
		batch = model.batches[b]
		model.newtontempbuf = OpenCL.Buffer(Float32, model.context, :rw, model.K^2 * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:newtongrad)
		quoteblock =
		quote
		batch = model.batches[b]
		model.newtongradbuf = OpenCL.Buffer(Float32, model.context, :rw, model.K * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:newtoninvhess)
		quoteblock =
		quote
		batch = model.batches[b]
		model.newtoninvhessbuf = OpenCL.Buffer(Float32, model.context, :rw, model.K^2 * (length(batch) + 64 - length(batch) % 64))	
		end

	elseif expr.args[2] == :(:mu)
		quoteblock =
		quote
		model.mubuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.mu)
		end

	elseif expr.args[2] == :(:sigma)
		quoteblock =
		quote
		model.sigmabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.sigma)
		end

	elseif expr.args[2] == :(:invsigma)
		quoteblock =
		quote
		model.invsigmabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.invsigma)
		end

	elseif expr.args[2] == :(:lambda)
		quoteblock =
		quote
		model.lambdabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.lambda..., zeros(Float32, model.K, 64 - model.M % 64)))
		end

	elseif expr.args[2] == :(:vsq)
		quoteblock =
		quote
		model.vsqbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.vsq..., zeros(Float32, model.K, 64 - model.M % 64)))
		end

	elseif expr.args[2] == :(:lzeta)
		quoteblock =
		quote
		model.lzetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.lzeta)
		end

	elseif expr.args[2] == :(:alef)
		quoteblock = 
		quote
		model.alefbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alef)
		end

	elseif expr.args[2] == :(:newalef)
		quoteblock = 
		quote
		model.newalefbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=fill(model.a, model.K, model.V))
		end

	elseif expr.args[2] == :(:bet)
		quoteblock =
		quote
		model.betbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.bet)
		end

	elseif expr.args[2] == :(:gimel)
		quoteblock =
		quote
		model.gimelbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gimel..., zeros(Float32, model.K, 64 - model.M % 64)))
		end

	elseif expr.args[2] == :(:dalet)
		quoteblock =
		quote
		model.daletbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.dalet)
		end

	elseif expr.args[2] == :(:he)
		quoteblock = 
		quote
		model.hebuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.he)
		end

	elseif expr.args[2] == :(:newhe)
		quoteblock = 
		quote
		model.newhebuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=fill(model.e, model.K, model.U))
		end

	elseif expr.args[2] == :(:vav)
		quoteblock =
		quote
		model.vavbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.vav)
		end

	elseif expr.args[2] == :(:zayin)
		quoteblock =
		quote
		model.zayinbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.zayin..., zeros(Float32, model.K, 64 - model.M % 64)))
		end

	elseif expr.args[2] == :(:het)
		quoteblock =
		quote
		model.hetbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.het)
		end

	elseif expr.args[2] == :(:xi)
		quoteblock =
		quote
		batch = model.batches[b]
		model.xibuf = OpenCL.Buffer(Float32, model.context, :rw, 2model.K * (sum(model.R[batch]) + 64 - sum(model.R[batch]) % 64))
		end
	end
	return quoteblock
end

macro host(args...)
	if isa(args[1], Expr)
		expr = args[1]
	else
		b = args[1]
		expr = args[2]
	end

	if expr.args[2] == :(:alphabuf)
		quoteblock =
		quote
		model.alpha = OpenCL.read(model.queue, model.alphabuf)
		end

	elseif expr.args[2] == :(:betabuf)
		quoteblock = 
		quote
		model.beta = reshape(OpenCL.read(model.queue, model.betabuf), model.K, model.V)
		end

	elseif expr.args[2] == :(:gammabuf)
		quoteblock = 
		quote
		hostgamma = reshape(OpenCL.read(model.queue, model.gammabuf), model.K, model.M + 64 - model.M % 64)
		model.gamma = [hostgamma[:,d] for d in 1:model.M]
		end

	elseif expr.args[2] == :(:phibuf)
		quoteblock = 
		quote
		batch = model.batches[b]
		Npsums = model.Npsums[b]
		hostphi = reshape(OpenCL.read(model.queue, model.phibuf), model.K, sum(model.N[batch]) + 64 - sum(model.N[batch]) % 64)
		model.phi = [hostphi[:,Npsums[d]+1:Npsums[d+1]] for d in 1:length(batch)]
		end

	elseif expr.args[2] == :(:Elogthetabuf)
		quoteblock = 
		quote
		batch = model.batches[b]
		hostElogtheta = reshape(OpenCL.read(model.queue, model.Elogthetabuf), model.K, length(batch) + 64 - length(batch) % 64)
		model.Elogtheta = [hostElogtheta[:,d] for d in 1:length(batch)]
		end

	elseif expr.args[2] == :(:Elogthetasumbuf)
		quoteblock = 
		quote
		model.Elogthetasum = OpenCL.read(model.queue, model.Elogthetasumbuf)
		end

	elseif expr.args[2] == :(:mubuf)
		quoteblock =
		quote
		model.mu = OpenCL.read(model.queue, model.mubuf)
		end

	elseif expr.args[2] == :(:sigmabuf)
		quoteblock =
		quote
		model.sigma = reshape(OpenCL.read(model.queue, model.sigmabuf), model.K, model.K)
		end

	elseif expr.args[2] == :(:invsigmabuf)
		quoteblock =
		quote
		model.invsigma = reshape(OpenCL.read(model.queue, model.invsigmabuf), model.K, model.K)
		end

	elseif expr.args[2] == :(:lambdabuf)
		quoteblock = 
		quote
		hostlambda = reshape(OpenCL.read(model.queue, model.lambdabuf), model.K, model.M + 64 - model.M % 64)
		model.lambda = [hostlambda[:,d] for d in 1:model.M]
		end

	elseif expr.args[2] == :(:vsqbuf)
		quoteblock = 
		quote
		hostvsq = reshape(OpenCL.read(model.queue, model.vsqbuf), model.K, model.M + 64 - model.M % 64)
		model.vsq = [hostvsq[:,d] for d in 1:model.M]
		end

	elseif expr.args[2] == :(:lzetabuf)
		quoteblock = 
		quote
		model.lzeta = OpenCL.read(model.queue, model.lzetabuf)
		end

	elseif expr.args[2] == :(:alefbuf)
		quoteblock = 
		quote
		model.alef = reshape(OpenCL.read(model.queue, model.alefbuf), model.K, model.V)
		end

	elseif expr.args[2] == :(:betbuf)
		quoteblock = 
		quote
		model.bet = OpenCL.read(model.queue, model.betbuf)
		end

	elseif expr.args[2] == :(:gimelbuf)
		quoteblock = 
		quote
		hostgimel = reshape(OpenCL.read(model.queue, model.gimelbuf), model.K, model.M + 64 - model.M % 64)
		model.gimel = [hostgimel[:,d] for d in 1:model.M]
		end

	elseif expr.args[2] == :(:daletbuf)
		quoteblock = 
		quote
		model.dalet = OpenCL.read(model.queue, model.daletbuf)
		end

	elseif expr.args[2] == :(:hebuf)
		quoteblock = 
		quote
		model.he = reshape(OpenCL.read(model.queue, model.hebuf), model.K, model.U)
		end

	elseif expr.args[2] == :(:vavbuf)
		quoteblock = 
		quote
		model.vav = OpenCL.read(model.queue, model.vavbuf)
		end

	elseif expr.args[2] == :(:zayinbuf)
		quoteblock = 
		quote
		hostzayin = reshape(OpenCL.read(model.queue, model.zayinbuf), model.K, model.M + 64 - model.M % 64)
		model.zayin = [hostzayin[:,d] for d in 1:model.M]
		end

	elseif expr.args[2] == :(:hetbuf)
		quoteblock = 
		quote
		model.het = OpenCL.read(model.queue, model.hetbuf)
		end

	elseif expr.args[2] == :(:xibuf)
		quoteblock = 
		quote
		batch = model.batches[b]
		Rpsums = model.Rpsums[b]
		hostxi = reshape(OpenCL.read(model.queue, model.xibuf), 2model.K, sum(model.R[batch]) + 64 - sum(model.R[batch]) % 64)
		model.xi = [hostxi[:,Rpsums[m]+1:Rpsums[m+1]] for m in 1:length(batch)]
		end
	end
	return quoteblock
end

macro gpu(args...)
	if length(args) == 1
		@assert isa(args[1], Expr)
		batchsize = Inf
		expr = args[1]
	elseif length(args) == 2
		@assert isa(args[1], Integer)
		@assert isa(args[2], Expr)
		batchsize = args[1]
		expr = args[2]
	else
		throw(ArgumentError("Wrong number of arguments."))
	end

	@assert ispositive(batchsize)
	@assert expr.args[1] == :train! "GPU acceleration only applies to the train! function."

	quote
	local model = $(esc(expr.args[2]))
	local batchsize = Int(min($(esc(batchsize)), length(model.corp)))
	local kwargs = [(kw.args[1], kw.args[2]) for kw in $(esc(expr.args[3:end]))]
	
	if isa(model, LDA)
		fakecorp = Corpus(docs=[Document([1])], lex=["1"])
		gpumodel = gpuLDA(fakecorp, 1)

		gpumodel.corp = model.corp

		gpumodel.K = model.K
		gpumodel.M = model.M
		gpumodel.V = model.V
		gpumodel.N = model.N
		gpumodel.C = model.C

		gpumodel.batches = partition(1:model.M, batchsize)
		gpumodel.B = length(gpumodel.batches)
		
		gpumodel.topics = model.topics

		gpumodel.alpha = model.alpha
		gpumodel.beta = model.beta
		gpumodel.gamma = model.gamma
		gpumodel.phi = unshift!([ones(model.K, model.N[d]) / model.K for d in gpumodel.batches[1][2:end]], model.phi)
		gpumodel.elbo = model.elbo
		
		fixmodel!(gpumodel)
		train!(gpumodel; kwargs...)
		
		model.topics = gpumodel.topics
		model.alpha = gpumodel.alpha
		model.beta = gpumodel.beta
		model.gamma = gpumodel.gamma
		model.phi = gpumodel.phi[1]
		model.Elogtheta = gpumodel.Elogtheta[1]
		model.Elogthetasum = zeros(gpumodel.K)
		model.elbo = gpumodel.elbo

		model.beta ./= sum(model.beta, 2)
		model.phi ./= sum(model.phi, 1)
		nothing

	elseif isa(model, fLDA)
		nothing

	elseif isa(model, CTM)
		fakecorp = Corpus(docs=[Document([1])], lex=["1"])
		gpumodel = gpuCTM(fakecorp, 1)

		gpumodel.corp = model.corp

		gpumodel.K = model.K
		gpumodel.M = model.M
		gpumodel.V = model.V
		gpumodel.N = model.N
		gpumodel.C = model.C

		gpumodel.batches = partition(1:model.M, batchsize)
		gpumodel.B = length(gpumodel.batches)
		
		gpumodel.topics = model.topics

		gpumodel.mu = model.mu
		gpumodel.sigma = model.sigma
		gpumodel.beta = model.beta
		gpumodel.lambda = model.lambda
		gpumodel.vsq = model.vsq
		gpumodel.lzeta = fill(model.lzeta, model.M)
		gpumodel.phi = unshift!([ones(model.K, model.N[d]) / model.K for d in gpumodel.batches[1][2:end]], model.phi)
		gpumodel.elbo = model.elbo
		
		fixmodel!(gpumodel)
		train!(gpumodel; kwargs...)
		
		model.topics = gpumodel.topics
		model.mu = gpumodel.mu
		model.sigma = gpumodel.sigma
		model.invsigma = gpumodel.invsigma
		model.beta = gpumodel.beta
		model.lambda = gpumodel.lambda
		model.vsq = gpumodel.vsq
		model.lzeta = gpumodel.lzeta[1]
		model.phi = gpumodel.phi[1]
		model.elbo = gpumodel.elbo

		model.beta ./= sum(model.beta, 2)
		model.phi ./= sum(model.phi, 1)
		nothing

	elseif isa(model, fCTM)
		nothing

	elseif isa(model, DTM)
		nothing

	elseif isa(model, CTPF)
		fakecorp = Corpus(docs=[Document([1], readers=[1])], lex=["1"], users=["1"])
		gpumodel = gpuCTPF(fakecorp, 1)

		gpumodel.corp = model.corp

		gpumodel.batches = partition(1:model.M, batchsize)
		gpumodel.B = length(gpumodel.batches)

		gpumodel.topics = model.topics
		gpumodel.scores = model.scores
		gpumodel.libs = model.libs
		gpumodel.drecs = model.drecs
		model.urecs = model.urecs

		gpumodel.K = model.K
		gpumodel.M = model.M
		gpumodel.V = model.V
		gpumodel.U = model.U
		gpumodel.N = model.N
		gpumodel.C = model.C
		gpumodel.R = model.R

		gpumodel.a = model.a
		gpumodel.b = model.b
		gpumodel.c = model.c
		gpumodel.d = model.d
		gpumodel.e = model.e
		gpumodel.f = model.f
		gpumodel.g = model.g
		gpumodel.h = model.h
		gpumodel.alef = model.alef
		gpumodel.bet = model.bet
		gpumodel.gimel = model.gimel
		gpumodel.dalet = model.dalet
		gpumodel.he = model.he
		gpumodel.vav = model.vav
		gpumodel.zayin = model.zayin
		gpumodel.het = model.het
		gpumodel.phi = unshift!([ones(model.K, model.N[d]) / model.K for d in gpumodel.batches[1][2:end]], model.phi)
		gpumodel.xi = unshift!([ones(2model.K, model.R[d]) / 2model.K for d in gpumodel.batches[1][2:end]], model.xi)
		gpumodel.elbo = model.elbo
		
		fixmodel!(gpumodel)
		train!(gpumodel; kwargs...)

		model.topics = gpumodel.topics
		model.scores = gpumodel.scores
		model.drecs = gpumodel.drecs
		model.urecs = gpumodel.urecs
		
		model.alef = gpumodel.alef
		model.bet = gpumodel.bet
		model.gimel = gpumodel.gimel
		model.dalet = gpumodel.dalet
		model.he = gpumodel.he
		model.vav = gpumodel.vav
		model.zayin = gpumodel.zayin
		model.het = gpumodel.het
		model.phi = gpumodel.phi[1]
		model.xi = gpumodel.xi[1]
		model.elbo = gpumodel.elbo

		model.phi ./= sum(model.phi, 1)
		model.xi ./= sum(model.xi, 1)
		nothing

	else
		nothing
	end
	end
end