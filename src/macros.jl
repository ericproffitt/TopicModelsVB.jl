### Macros for TopicModelsVB
### Eric Proffitt
### December 3, 2019

macro juliadots(str::String)
	"Print Julia dots before bolded string output."
	"For vanilla strings."

	expr = :(	
			print(Crayon(foreground=:red, bold=true), " ●");
			print(Crayon(foreground=:green, bold=true), "●");
			print(Crayon(foreground=:blue, bold=true), "● ");
			print(Crayon(foreground=:white, bold=true), $str);
			)
	
	return expr
end

macro juliadots(expr::Expr)
	"Print Julia dots before bolded string output."
	"For interpolated strings."

	expr = :(	
			print(Crayon(foreground=:red, bold=true), " ●");
			print(Crayon(foreground=:green, bold=true), "●");
			print(Crayon(foreground=:blue, bold=true), "● ");
			print(Crayon(foreground=:white, bold=true), :($($expr)))
			)
	
	return expr
end

macro boink(expr::Expr)
	"Add EPSILON to a numerical variable or array."

	expr = :(:($($expr)) .+ EPSILON)
	return expr
end

macro bumper(expr::Expr)
	"Add EPSILON to a numerical variable or array during variable assignment."

	if (expr.head == :.) || (expr.head == :ref)
		expr = :(:($($expr)) .+= EPSILON)
	
	elseif expr.head == :(=)
		expr = :(:($($(expr.args[1]))) = EPSILON .+ :($($(expr.args[2]))))
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

	model = expr.args[1]

	if expr.args[2] == :(:Npsums)
		quoteblock =
		quote
		$(esc(model)).Npsumsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).Npsums[$(esc(b))])
		end

	elseif expr.args[2] == :(:Jpsums)
		quoteblock =
		quote
		$(esc(model)).Jpsumsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).Jpsums[$(esc(b))])
		end

	elseif expr.args[2] == :(:Rpsums)
		quoteblock =
		quote
		$(esc(model)).Rpsumsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).Rpsums[$(esc(b))])
		end

	elseif expr.args[2] == :(:Ypsums)
		quoteblock =
		quote
		$(esc(model)).Ypsumsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).Ypsums[$(esc(b))])
		end

	elseif expr.args[2] == :(:terms)
		quoteblock =
		quote
		$(esc(model)).termsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).terms[$(esc(b))])
		end

	elseif expr.args[2] == :(:counts)
		quoteblock =
		quote
		$(esc(model)).countsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).counts[$(esc(b))])
		end

	elseif expr.args[2] == :(:words)
		quoteblock =
		quote
		$(esc(model)).wordsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).words[$(esc(b))])
		end

	elseif expr.args[2] == :(:readers)
		quoteblock =
		quote
		$(esc(model)).readersbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).readers[$(esc(b))])
		end

	elseif expr.args[2] == :(:ratings)
		quoteblock =
		quote
		$(esc(model)).ratingsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).ratings[$(esc(b))])
		end

	elseif expr.args[2] == :(:views)
		quoteblock =
		quote
		$(esc(model)).viewsbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).views[$(esc(b))])
		end

	elseif expr.args[2] == :(:alpha)
		quoteblock =
		quote
		$(esc(model)).alphabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).alpha)
		end

	elseif expr.args[2] == :(:beta)
		quoteblock =
		quote
		$(esc(model)).betabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).beta)
		end

	elseif expr.args[2] == :(:newbeta)
		quoteblock =
		quote
		$(esc(model)).newbetabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=zeros(Float32, $(esc(model)).K, $(esc(model)).V))
		end

	elseif expr.args[2] == :(:gamma)
		quoteblock =
		quote
		$(esc(model)).gammabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).gamma..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))
		end

	elseif expr.args[2] == :(:phi)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).phibuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=ones(Float32, $(esc(model)).K, sum($(esc(model)).N[batch]) + 64 - sum($(esc(model)).N[batch]) % 64) / $(esc(model)).K)
		end

	elseif expr.args[2] == :(:Elogtheta)
		quoteblock = 
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).Elogthetabuf = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:Elogthetasum)
		quoteblock =
		quote
		$(esc(model)).Elogthetasumbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).Elogthetasum)
		end

	elseif expr.args[2] == :(:C)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).Cbuf = cl.Buffer(Int, $(esc(model)).context, (:r, :copy), hostbuf=$(esc(model)).C[batch])
		end

	elseif expr.args[2] == :(:newtontemp)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).newtontempbuf = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K^2 * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:newtongrad)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).newtongradbuf = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K * (length(batch) + 64 - length(batch) % 64))
		end

	elseif expr.args[2] == :(:newtoninvhess)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).newtoninvhessbuf = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K^2 * (length(batch) + 64 - length(batch) % 64))	
		end

	elseif expr.args[2] == :(:mu)
		quoteblock =
		quote
		$(esc(model)).mubuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).mu)
		end

	elseif expr.args[2] == :(:sigma)
		quoteblock =
		quote
		$(esc(model)).sigmabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).sigma)
		end

	elseif expr.args[2] == :(:invsigma)
		quoteblock =
		quote
		$(esc(model)).invsigmabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).invsigma)
		end

	elseif expr.args[2] == :(:lambda)
		quoteblock =
		quote
		$(esc(model)).lambdabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).lambda..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))
		end

	elseif expr.args[2] == :(:vsq)
		quoteblock =
		quote
		$(esc(model)).vsqbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).vsq..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))
		end

	elseif expr.args[2] == :(:lzeta)
		quoteblock =
		quote
		$(esc(model)).lzetabuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).lzeta)
		end

	elseif expr.args[2] == :(:alef)
		quoteblock = 
		quote
		$(esc(model)).alefbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).alef)
		end

	elseif expr.args[2] == :(:newalef)
		quoteblock = 
		quote
		$(esc(model)).newalefbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=fill($(esc(model)).a, $(esc(model)).K, $(esc(model)).V))
		end

	elseif expr.args[2] == :(:bet)
		quoteblock =
		quote
		$(esc(model)).betbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).bet)
		end

	elseif expr.args[2] == :(:gimel)
		quoteblock =
		quote
		$(esc(model)).gimelbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).gimel..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))
		end

	elseif expr.args[2] == :(:dalet)
		quoteblock =
		quote
		$(esc(model)).daletbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).dalet)
		end

	elseif expr.args[2] == :(:he)
		quoteblock = 
		quote
		$(esc(model)).hebuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).he)
		end

	elseif expr.args[2] == :(:newhe)
		quoteblock = 
		quote
		$(esc(model)).newhebuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=fill($(esc(model)).e, $(esc(model)).K, $(esc(model)).U))
		end

	elseif expr.args[2] == :(:vav)
		quoteblock =
		quote
		$(esc(model)).vavbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).vav)
		end

	elseif expr.args[2] == :(:zayin)
		quoteblock =
		quote
		$(esc(model)).zayinbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).zayin..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))
		end

	elseif expr.args[2] == :(:het)
		quoteblock =
		quote
		$(esc(model)).hetbuf = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).het)
		end

	elseif expr.args[2] == :(:xi)
		quoteblock =
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		$(esc(model)).xibuf = cl.Buffer(Float32, $(esc(model)).context, :rw, 2 * $(esc(model)).K * (sum($(esc(model)).R[batch]) + 64 - sum($(esc(model)).R[batch]) % 64))
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

	model = expr.args[1]

	if expr.args[2] == :(:alphabuf)
		quoteblock =
		quote
		$(esc(model)).alpha = cl.read($(esc(model)).queue, $(esc(model)).alphabuf)
		end

	elseif expr.args[2] == :(:betabuf)
		quoteblock = 
		quote
		$(esc(model)).beta = reshape(cl.read($(esc(model)).queue, $(esc(model)).betabuf), $(esc(model)).K, $(esc(model)).V)
		end

	elseif expr.args[2] == :(:gammabuf)
		quoteblock = 
		quote
		hostgamma = reshape(cl.read($(esc(model)).queue, $(esc(model)).gammabuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).gamma = [hostgamma[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:phibuf)
		quoteblock = 
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		Npsums = $(esc(model)).Npsums[$(esc(b))]
		hostphi = reshape(cl.read($(esc(model)).queue, $(esc(model)).phibuf), $(esc(model)).K, sum($(esc(model)).N[batch]) + 64 - sum($(esc(model)).N[batch]) % 64)
		$(esc(model)).phi = [hostphi[:,Npsums[d]+1:Npsums[d+1]] for d in 1:length(batch)]
		end

	elseif expr.args[2] == :(:Elogthetabuf)
		quoteblock = 
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		hostElogtheta = reshape(cl.read($(esc(model)).queue, $(esc(model)).Elogthetabuf), $(esc(model)).K, length(batch) + 64 - length(batch) % 64)
		$(esc(model)).Elogtheta = [hostElogtheta[:,d] for d in 1:length(batch)]
		end

	elseif expr.args[2] == :(:Elogthetasumbuf)
		quoteblock = 
		quote
		$(esc(model)).Elogthetasum = cl.read($(esc(model)).queue, $(esc(model)).Elogthetasumbuf)
		end

	elseif expr.args[2] == :(:mubuf)
		quoteblock =
		quote
		$(esc(model)).mu = cl.read($(esc(model)).queue, $(esc(model)).mubuf)
		end

	elseif expr.args[2] == :(:sigmabuf)
		quoteblock =
		quote
		$(esc(model)).sigma = reshape(cl.read($(esc(model)).queue, $(esc(model)).sigmabuf), $(esc(model)).K, $(esc(model)).K)
		end

	elseif expr.args[2] == :(:invsigmabuf)
		quoteblock =
		quote
		$(esc(model)).invsigma = reshape(cl.read($(esc(model)).queue, $(esc(model)).invsigmabuf), $(esc(model)).K, $(esc(model)).K)
		end

	elseif expr.args[2] == :(:lambdabuf)
		quoteblock = 
		quote
		hostlambda = reshape(cl.read($(esc(model)).queue, $(esc(model)).lambdabuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).lambda = [hostlambda[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:vsqbuf)
		quoteblock = 
		quote
		hostvsq = reshape(cl.read($(esc(model)).queue, $(esc(model)).vsqbuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).vsq = [hostvsq[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:lzetabuf)
		quoteblock = 
		quote
		$(esc(model)).lzeta = cl.read($(esc(model)).queue, $(esc(model)).lzetabuf)
		end

	elseif expr.args[2] == :(:alefbuf)
		quoteblock = 
		quote
		$(esc(model)).alef = reshape(cl.read($(esc(model)).queue, $(esc(model)).alefbuf), $(esc(model)).K, $(esc(model)).V)
		end

	elseif expr.args[2] == :(:betbuf)
		quoteblock = 
		quote
		$(esc(model)).bet = cl.read($(esc(model)).queue, $(esc(model)).betbuf)
		end

	elseif expr.args[2] == :(:gimelbuf)
		quoteblock = 
		quote
		hostgimel = reshape(cl.read($(esc(model)).queue, $(esc(model)).gimelbuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).gimel = [hostgimel[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:daletbuf)
		quoteblock = 
		quote
		$(esc(model)).dalet = cl.read($(esc(model)).queue, $(esc(model)).daletbuf)
		end

	elseif expr.args[2] == :(:hebuf)
		quoteblock = 
		quote
		$(esc(model)).he = reshape(cl.read($(esc(model)).queue, $(esc(model)).hebuf), $(esc(model)).K, $(esc(model)).U)
		end

	elseif expr.args[2] == :(:vavbuf)
		quoteblock = 
		quote
		$(esc(model)).vav = cl.read($(esc(model)).queue, $(esc(model)).vavbuf)
		end

	elseif expr.args[2] == :(:zayinbuf)
		quoteblock = 
		quote
		hostzayin = reshape(cl.read($(esc(model)).queue, $(esc(model)).zayinbuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).zayin = [hostzayin[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:hetbuf)
		quoteblock = 
		quote
		$(esc(model)).het = cl.read($(esc(model)).queue, $(esc(model)).hetbuf)
		end

	elseif expr.args[2] == :(:xibuf)
		quoteblock = 
		quote
		batch = $(esc(model)).batches[$(esc(b))]
		Rpsums = $(esc(model)).Rpsums[$(esc(b))]
		hostxi = reshape(cl.read($(esc(model)).queue, $(esc(model)).xibuf), 2 * $(esc(model)).K, sum($(esc(model)).R[batch]) + 64 - sum($(esc(model)).R[batch]) % 64)
		$(esc(model)).xi = [hostxi[:,Rpsums[m]+1:Rpsums[m+1]] for m in 1:length(batch)]
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

