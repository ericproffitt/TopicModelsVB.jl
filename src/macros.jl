### Macros for TopicModelsVB
### Eric Proffitt
### December 3, 2019

macro juliadots(str::String)
	"Print Julia dots before bolded string output."
	"For vanilla strings."

	expr_out = :(	
				print(Crayon(foreground=:red, bold=true), " ●");
				print(Crayon(foreground=:green, bold=true), "●");
				print(Crayon(foreground=:blue, bold=true), "● ");
				print(Crayon(foreground=:white, bold=true), $str);
				)
	
	return expr_out
end

macro juliadots(expr::Expr)
	"Print Julia dots before bolded string output."
	"For interpolated strings."

	expr_out = :(	
				print(Crayon(foreground=:red, bold=true), " ●");
				print(Crayon(foreground=:green, bold=true), "●");
				print(Crayon(foreground=:blue, bold=true), "● ");
				print(Crayon(foreground=:white, bold=true), :($($expr)))
				)
	
	return expr_out
end

macro boink(expr::Expr)
	"Add EPSILON to a numerical variable or array."

	expr_out = :(:($($expr)) .+ EPSILON)
	return expr_out
end

macro positive(expr::Expr)
	"Add EPSILON to a numerical variable or array during variable assignment."

	if (expr.head == :.) || (expr.head == :ref)
		expr_out = :(:($($expr)) .+= EPSILON)
	
	elseif expr.head == :(=)
		expr_out = :(:($($(expr.args[1]))) = EPSILON .+ :($($(expr.args[2]))))
	end

	return expr_out
end

macro finite(expr::Expr)
	"Prevent overflow of floating point to Inf by returning floatmax() of value."

	if (expr.head == :.) || (expr.head == :ref)
		expr_out = :(:($($expr)) = sign.(:($($expr))) .* min.(abs.(:($($expr))), floatmax.(:($($expr)))))
	
	elseif expr.head == :(=)
		expr_out = :(:($($(expr.args[1]))) = sign.(:($($(expr.args[2])))) .* min.(abs.(:($($(expr.args[2])))), floatmax.(:($($(expr.args[2]))))))

	elseif expr.head == :(-=)
		expr_out = :(:($($(expr.args[1]))) = sign.(:($($(expr.args[1])))) .* min.(abs.(:($($(expr.args[1]))) - :($($(expr.args[2])))), floatmax.(:($($(expr.args[1]))) - :($($(expr.args[2]))))))
	end

	return expr_out
end

macro buffer(expr::Expr)
	"Load individual variable into buffer memory."

	model = expr.args[1]

	if expr.args[2] == :(:alpha)
		expr_out = :($(esc(model)).alpha_buffer = cl.Buffer(Float32, $(esc(model)).context, (:rw, :copy), hostbuf=$(esc(model)).alpha))

	elseif expr.args[2] == :(:invsigma)
		expr_out = :($(esc(model)).invsigma_buffer = cl.Buffer(Float32, $(esc(model)).context, (:r, :copy), hostbuf=Matrix($(esc(model)).invsigma)))
	end
	
	return expr_out
end

macro host(expr::Expr)
	"Load individual variable into host memory."

	model = expr.args[1]

	if expr.args[2] == :(:Elogtheta_sum_buffer)
		expr_out = :($(esc(model)).Elogtheta_sum = cl.read($(esc(model)).queue, $(esc(model)).Elogtheta_sum_buffer))

	elseif expr.args[2] == :(:Elogtheta_dist_buffer)
		expr_out = :($(esc(model)).Elogtheta_dist = cl.read($(esc(model)).queue, $(esc(model)).Elogtheta_dist_buffer)[1:$(esc(model)).M])

	elseif expr.args[2] == :(:sigma_buffer)
		expr_out = :($(esc(model)).sigma = Symmetric(reshape(cl.read($(esc(model)).queue, $(esc(model)).sigma_buffer), model.K, model.K)))

	elseif expr.args[2] == :(:lambda_dist_buffer)
		expr_out = :($(esc(model)).lambda_dist = cl.read($(esc(model)).queue, $(esc(model)).lambda_dist_buffer)[1:$(esc(model)).M])

	elseif expr.args[2] == :(:gimel_buffer)
		expr_out = 
		quote
		gimel_host = reshape(cl.read($(esc(model)).queue, $(esc(model)).gimel_buffer), $(esc(model)).K, $(esc(model)).M + 64 - $(esc(model)).M % 64)
		$(esc(model)).gimel = [gimel_host[:,d] for d in 1:$(esc(model)).M]
		end
	end

	return expr_out
end

macro gpu(expr::Expr)
	"Train model on GPU."

	expr.args[1] == :train! || throw(ArgumentError("GPU acceleration only applies to the train! function."))

	quote
		local model = $(esc(expr.args[2]))
		local kwargs = [(kw.args[1], kw.args[2]) for kw in $(esc(expr.args[3:end]))]
		
		if isa(model, LDA)
			gpumodel = gpuLDA(Corpus(), 1)
			gpumodel.corp = model.corp
			gpumodel.K = model.K
			gpumodel.M = model.M
			gpumodel.V = model.V
			gpumodel.N = model.N
			gpumodel.C = model.C
			gpumodel.topics = model.topics
			gpumodel.alpha = model.alpha
			gpumodel.beta = model.beta
			gpumodel.Elogtheta = model.Elogtheta
			gpumodel.gamma = model.gamma
			gpumodel.phi = [ones(model.K, model.N[d]) / model.K for d in 1:model.M]
			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)
			
			model.topics = gpumodel.topics
			model.alpha = gpumodel.alpha
			model.beta = gpumodel.beta
			model.Elogtheta = gpumodel.Elogtheta
			model.gamma = gpumodel.gamma
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.beta ./= sum(model.beta, dims=2)
			model.phi ./= sum(model.phi, dims=1)
			nothing

		elseif isa(model, fLDA)
			nothing

		elseif isa(model, CTM)
			gpumodel = gpuCTM(Corpus(), 1)
			gpumodel.corp = model.corp
			gpumodel.K = model.K
			gpumodel.M = model.M
			gpumodel.V = model.V
			gpumodel.N = model.N
			gpumodel.C = model.C
			gpumodel.topics = model.topics
			gpumodel.mu = model.mu
			gpumodel.sigma = model.sigma
			gpumodel.invsigma = model.invsigma
			gpumodel.beta = model.beta
			gpumodel.lambda = model.lambda
			gpumodel.vsq = model.vsq
			gpumodel.logzeta = model.logzeta
			gpumodel.phi = [ones(model.K, model.N[d]) / model.K for d in 1:model.M]
			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)
			
			model.topics = gpumodel.topics
			model.mu = gpumodel.mu
			model.sigma = gpumodel.sigma
			model.invsigma = gpumodel.invsigma
			model.beta = gpumodel.beta
			model.lambda = gpumodel.lambda
			model.vsq = gpumodel.vsq
			model.logzeta = gpumodel.logzeta
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.beta ./= sum(model.beta, dims=2)
			model.phi ./= sum(model.phi, dims=1)
			nothing

		elseif isa(model, fCTM)
			nothing

		elseif isa(model, CTPF)
			gpumodel = gpuCTPF(Corpus(), 1)
			gpumodel.corp = model.corp
			gpumodel.topics = model.topics
			gpumodel.scores = model.scores
			gpumodel.libs = model.libs
			gpumodel.drecs = model.drecs
			gpumodel.urecs = model.urecs
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
			gpumodel.he = model.he
			gpumodel.bet = model.bet
			gpumodel.vav = model.vav
			gpumodel.gimel = model.gimel
			gpumodel.zayin = model.zayin
			gpumodel.dalet = model.dalet
			gpumodel.het = model.het
			gpumodel.phi = [ones(model.K, model.N[d]) / model.K for d in 1:model.M]
			gpumodel.xi = [ones(2model.K, model.R[d]) / 2model.K for d in 1:model.M]
			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)

			model.topics = gpumodel.topics
			model.scores = gpumodel.scores
			model.drecs = gpumodel.drecs
			model.urecs = gpumodel.urecs
			model.alef = gpumodel.alef
			model.he = gpumodel.he
			model.bet = gpumodel.bet
			model.vav = gpumodel.vav
			model.gimel = gpumodel.gimel
			model.zayin = gpumodel.zayin
			model.dalet = gpumodel.dalet
			model.het = gpumodel.het
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.xi = gpumodel.xi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.phi ./= sum(model.phi, dims=1)
			model.xi ./= sum(model.xi, dims=1)
			nothing

		else
			nothing
		end
	end
end

