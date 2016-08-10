macro juliadots(expr::Expr)
	expr = :(print_with_color(:red, " ●");
				print_with_color(:green, "●");
				print_with_color(:blue, "● ");
				print_with_color(:bold, $expr))
	return expr
end

macro boink(expr::Expr)
	expr = :($expr + epsln)
	return expr
end

macro bumper(expr::Expr)
	if expr.head == :.
		expr = :($expr += epsln)
	elseif expr.head == :(=)
		expr = :($(expr.args[1]) = epsln + $(expr.args[2]))
	end
	return expr
end

macro buf(expr::Expr)
	if expr.args[2] == :(:alpha)
		quoteblock =
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.alphabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.alpha)
		end
		end

	elseif expr.args[2] == :(:Elogthetasum)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.Elogthetasumbuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.Elogthetasum)
		end
		end
	end
	return quoteblock
end

macro host(expr::Expr)
	if expr.args[2] == :(:alphabuf)
		quoteblock =
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.alpha = OpenCL.read(model.queue, model.alphabuf)
		end
		end

	elseif expr.args[2] == :(:Elogthetasumbuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.Elogthetasum = OpenCL.read(model.queue, model.Elogthetasumbuf)
		end
		end
	end
	return quoteblock
end

macro gpu(expr::Expr)
	@assert expr.args[1] == :train! "GPU acceleration only applies to the train! function."

	quote
	local model = $(esc(expr.args[2]))
	local kwargs = [(kw.args[1], kw.args[2]) for kw in $(esc(expr.args[3:end]))]
	
	if isa(model, LDA)
		fakecorp = Corpus(docs=[Document([1])], lex=["1"])
		gpumodel = gpuLDA(fakecorp, 1)

		gpumodel.corp = model.corp

		gpumodel.K = model.K
		gpumodel.M, gpumodel.V = size(gpumodel.corp)[1:2]
		gpumodel.N = [length(doc) for doc in gpumodel.corp]
		gpumodel.C = [size(doc) for doc in gpumodel.corp]

		gpumodel.topics = [collect(1:gpumodel.V) for _ in 1:gpumodel.K]

		gpumodel.alpha = model.alpha
		gpumodel.beta = model.beta
		gpumodel.gamma = model.gamma
		gpumodel.phi = unshift!([ones(model.K, model.N[d]) / model.K for d in 2:model.M], model.phi)
		gpumodel.elbo = model.elbo
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
		nothing

	elseif isa(model, fCTM)
		nothing

	elseif isa(model, DTM)
		nothing

	elseif isa(model, CTPF)
		gpumodel = gpuCTPF(model.corp, model.K)

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
		gpumodel.phi = model.phi
		gpumodel.xi = model.xi
		gpumodel.elbo = model.elbo
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
		model.phi = gpumodel.phi
		model.xi = gpumodel.xi
		model.elbo = gpumodel.elbo

		for d in 1:model.M
			model.phi[d] ./= sum(model.phi[d], 1)
			model.xi[d] ./= sum(model.xi[d], 1)
		end
		nothing

	else
		nothing
	end
	end
end