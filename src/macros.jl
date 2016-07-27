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

	elseif expr.args[2] == :(:beta)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.betabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.beta)
		end
		end

	elseif expr.args[2] == :(:gamma)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.gammabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.gamma...))
		end
		end

	elseif expr.args[2] == :(:phi)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.phibuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.phi...))
		end
		end
	
	elseif expr.args[2] == :(:Elogtheta)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.Elogthetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=hcat(model.Elogtheta...))
		end
		end

	elseif expr.args[2] == :(:SUMElogtheta)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.SUMElogthetabuf = OpenCL.Buffer(Float32, model.context, (:rw, :copy), hostbuf=model.SUMElogtheta)
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

	elseif expr.args[2] == :(:betabuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.beta = reshape(OpenCL.read(model.queue, model.betabuf), model.K, model.V)
		end
		end

	elseif expr.args[2] == :(:gammabuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			hostgamma = reshape(OpenCL.read(model.queue, model.gammabuf), model.K, model.M)
			model.gamma = [hostgamma[:,d] for d in 1:model.M]
		end
		end

	elseif expr.args[2] == :(:phibuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			Npsums = OpenCL.read(model.queue, model.Npsums)
			hostphi = reshape(OpenCL.read(model.queue, model.phibuf), model.K, sum(model.N))
			model.phi = [hostphi[:,Npsums[d]+1:Npsums[d+1]] for d in 1:model.M]	
		end
		end
	
	elseif expr.args[2] == :(:Elogthetabuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			hostElogtheta = reshape(OpenCL.read(model.queue, model.Elogthetabuf), model.K, model.M)
			model.Elogtheta = [hostElogtheta[:,d] for d in 1:model.M]	
		end
		end
	
	elseif expr.args[2] == :(:SUMElogthetabuf)
		quoteblock = 
		quote
		if isa($(esc(expr.args[1])), gpuLDA)
			model.SUMElogtheta = OpenCL.read(model.queue, model.SUMElogthetabuf)
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
		local gpumodel = gpuLDA(model.corp, model.K)

		gpumodel.alpha = model.alpha
		gpumodel.beta = model.beta
		gpumodel.phi = model.phi
		gpumodel.topics = model.topics
		train!(gpumodel; kwargs...)

		model.alpha = gpumodel.alpha
		model.beta = gpumodel.beta
		model.gamma = gpumodel.gamma
		model.phi = gpumodel.phi
		model.topics = gpumodel.topics

		model.beta ./= sum(model.beta, 2)
		for d in 1:model.M
			model.phi[d] ./= sum(model.phi[d], 1)
		end
		nothing

	else
		error("Model does not have GPU support.")
	end
	end
end