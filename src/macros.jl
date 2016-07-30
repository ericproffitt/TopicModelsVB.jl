macro juliadots(expr::Expr)
	expr = :(print_with_color(:red, " ●");
				print_with_color(:green, "●");
				print_with_color(:blue, "● ");
				print_with_color(:bold, $expr))
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

	elseif expr.args[2] == :(:sumElogtheta)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.sumElogthetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.sumElogtheta)
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

	elseif expr.args[2] == :(:sumElogthetabuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.sumElogtheta = OpenCL.read(model.queue, model.sumElogthetabuf)
		end
		end
	end
	return quoteblock
end

macro mem(expr::Expr)
	if expr.args[1] == :LDA
		quoteblock = 
		quote
		local corp = $(esc(expr.args[2]))
		local K = $(esc(expr.args[3]))		
		memLDA(corp, K)
		end

	elseif expr.args[1] == :fLDA
		quoteblock = 
		quote
		local corp = $(esc(expr.args[2]))
		local K = $(esc(expr.args[3]))		
		memfLDA(corp, K)
		end

	elseif expr.args[1] == :CTM
		quoteblock = 
		quote
		local corp = $(esc(expr.args[2]))
		local K = $(esc(expr.args[3]))		
		memCTM(corp, K)
		end

	elseif expr.args[1] == :fCTM
		quoteblock = 
		quote
		local corp = $(esc(expr.args[2]))
		local K = $(esc(expr.args[3]))		
		memfCTM(corp, K)
		end

	else
		error("Model does not have low memory support.")
	end
	return quoteblock
end

macro gpu(expr::Expr)
	@assert expr.args[1] == :train! "GPU acceleration only applies to the train! function."

	quote
	local model = $(esc(expr.args[2]))
	local kwargs = [(kw.args[1], kw.args[2]) for kw in $(esc(expr.args[3:end]))]
	
	if isa(model, LDA)
		gpumodel = gpuLDA(model.corp, model.K)

		gpumodel.alpha = model.alpha
		gpumodel.beta = model.beta
		gpumodel.phi = model.phi
		gpumodel.elbo = model.elbo
		train!(gpumodel; kwargs...)
		
		model.topics = gpumodel.topics
		model.alpha = gpumodel.alpha
		model.beta = gpumodel.beta
		model.gamma = gpumodel.gamma
		model.phi = gpumodel.phi
		model.Elogtheta = gpumodel.Elogtheta
		model.elbo = gpumodel.elbo

		model.beta ./= sum(model.beta, 2)
		for d in 1:model.M
			model.phi[d] ./= sum(model.phi[d], 1)
		end
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
		error("Model does not have GPU support.")
	end
	end
end