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

macro buffer(args...)
	"Load individual variables into buffer memory."

	expr = args[1]
	model = expr.args[1]

	if expr.args[2] == :(:alpha)
		expr_out = :($(esc(model)).alpha_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).alpha))

	elseif expr.args[2] == :(:newtontemp)
		expr_out = :($(esc(model)).newtontemp_buffer = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K^2 * (length(batch) + 64 - length(batch) % 64)))

	elseif expr.args[2] == :(:newtongrad)
		expr_out = :($(esc(model)).newtongrad_buffer = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K * (length(batch) + 64 - length(batch) % 64)))

	elseif expr.args[2] == :(:newtoninvhess)
		expr_out = :($(esc(model)).newtoninvhess_buffer = cl.Buffer(Float32, $(esc(model)).context, :rw, $(esc(model)).K^2 * (length(batch) + 64 - length(batch) % 64)))

	elseif expr.args[2] == :(:mu)
		expr_out = :($(esc(model)).mu_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).mu))

	elseif expr.args[2] == :(:sigma)
		expr_out = :($(esc(model)).sigma_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).sigma))

	elseif expr.args[2] == :(:invsigma)
		expr_out = :($(esc(model)).invsigma_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).invsigma))

	elseif expr.args[2] == :(:lambda)
		expr_out = :($(esc(model)).lambda_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).lambda..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64))))

	elseif expr.args[2] == :(:vsq)
		expr_out = $(esc(model)).vsq_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).vsq..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64)))

	elseif expr.args[2] == :(:lzeta)
		expr_out = :($(esc(model)).lzeta_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).lzeta))

	elseif expr.args[2] == :(:alef)
		expr_out = :($(esc(model)).alef_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).alef))

	elseif expr.args[2] == :(:bet)
		expr_out = :($(esc(model)).bet_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).bet))

	elseif expr.args[2] == :(:gimel)
		expr_out = :($(esc(model)).gimel_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).gimel..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64))))

	elseif expr.args[2] == :(:dalet)
		expr_out = :($(esc(model)).dalet_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).dalet))

	elseif expr.args[2] == :(:he)
		expr_out = :($(esc(model)).he_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).he))

	elseif expr.args[2] == :(:vav)
		expr_out = :($(esc(model)).vav_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).vav))

	elseif expr.args[2] == :(:zayin)
		expr_out = :($(esc(model)).zayin_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=hcat($(esc(model)).zayin..., zeros(Float32, $(esc(model)).K, 64 - $(esc(model)).M % 64))))

	elseif expr.args[2] == :(:het)
		expr_out = :($(esc(model)).het_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).het))

	elseif expr.args[2] == :(:xi)
		expr_out = :($(esc(model)).xi_buffer = cl.Buffer(Float32, $(esc(model)).context, :rw, 2 * $(esc(model)).K * (sum($(esc(model)).R) + 64 - sum($(esc(model)).R) % 64))
	end
	
	return expr_out
end

macro host(args...)
	"Load individual variables into host memory."

	expr = args[1]
	model = expr.args[1]

	if expr.args[2] == :(:Elogtheta_buffer)
		expr_out = 
		quote
		Elogtheta_host = reshape(cl.read($(esc(model)).queue, $(esc(model)).Elogtheta_buffer), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).Elogtheta = [hostElogtheta[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:mu_buffer)
		quoteblock =
		quote
		$(esc(model)).mu = cl.read($(esc(model)).queue, $(esc(model)).mubuf)
		end

	elseif expr.args[2] == :(:sigma_buffer)
		quoteblock =
		quote
		$(esc(model)).sigma = reshape(cl.read($(esc(model)).queue, $(esc(model)).sigmabuf), $(esc(model)).K, $(esc(model)).K)
		end

	elseif expr.args[2] == :(:invsigma_buffer)
		quoteblock =
		quote
		$(esc(model)).invsigma = reshape(cl.read($(esc(model)).queue, $(esc(model)).invsigmabuf), $(esc(model)).K, $(esc(model)).K)
		end

	elseif expr.args[2] == :(:lambda_buffer)
		quoteblock = 
		quote
		hostlambda = reshape(cl.read($(esc(model)).queue, $(esc(model)).lambdabuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).lambda = [hostlambda[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:vsq_buffer)
		quoteblock = 
		quote
		hostvsq = reshape(cl.read($(esc(model)).queue, $(esc(model)).vsqbuf), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).vsq = [hostvsq[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:lzeta_buffer)
		quoteblock = 
		quote
		$(esc(model)).lzeta = cl.read($(esc(model)).queue, $(esc(model)).lzetabuf)
		end

	elseif expr.args[2] == :(:alef_buffer)
		quoteblock = 
		quote
		$(esc(model)).alef = reshape(cl.read($(esc(model)).queue, $(esc(model)).alefbuf), $(esc(model)).K, $(esc(model)).V)
		end

	elseif expr.args[2] == :(:bet_buffer)
		quoteblock = 
		quote
		$(esc(model)).bet = cl.read($(esc(model)).queue, $(esc(model)).betbuf)
		end

	elseif expr.args[2] == :(:gimel_buffer)
		expr_out = 
		quote
		gimel_host = reshape(cl.read($(esc(model)).queue, $(esc(model)).gimel_buffer), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).gimel = [gimel_host[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:dalet_buffer)
		expr_out = :($(esc(model)).dalet = cl.read($(esc(model)).queue, $(esc(model)).dalet_buffer))

	elseif expr.args[2] == :(:he_buffer)
		expr_out = :($(esc(model)).he = reshape(cl.read($(esc(model)).queue, $(esc(model)).he_buffer), $(esc(model)).K, $(esc(model)).U))

	elseif expr.args[2] == :(:vav_buffer)
		expr_out = :($(esc(model)).vav = cl.read($(esc(model)).queue, $(esc(model)).vav_buffer))

	elseif expr.args[2] == :(:zayin_buffer)
		expr_out = 
		quote
		zayin_host = reshape(cl.read($(esc(model)).queue, $(esc(model)).zayin_buffer), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).zayin = [zayin_host[:,d] for d in 1:$(esc(model)).M]
		end

	elseif expr.args[2] == :(:het_buffer)
		expr_out = :($(esc(model)).het = cl.read($(esc(model)).queue, $(esc(model)).het_buffer))

	elseif expr.args[2] == :(:xi_buffer)
		expr_out = 
		quote
		hostxi = reshape(cl.read($(esc(model)).queue, $(esc(model)).xi_buffer), 2 * $(esc(model)).K, sum($(esc(model)).R[batch]) + 64 - sum($(esc(model)).R[batch]) % 64)
		$(esc(model)).xi = [hostxi[:,Rpsums[m]+1:Rpsums[m+1]] for m in 1:$(esc(model)).M]
		end
	end

	return expr_out
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

