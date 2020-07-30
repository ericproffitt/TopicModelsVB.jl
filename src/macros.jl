macro juliadots(str::String)
	"Print Julia dots before bolded string output."
	"For vanilla strings."

	expr_out = :(	
				print(Crayon(foreground=:red, bold=true), " ●");
				print(Crayon(foreground=:green, bold=true), "●");
				print(Crayon(foreground=:blue, bold=true), "● ");
				print(Crayon(foreground=:white, bold=true), $str);
				)
	
	return esc(expr_out)
end

macro juliadots(expr::Expr)
	"Print Julia dots before bolded string output."
	"For interpolated strings."

	expr_out = :(	
				print(Crayon(foreground=:red, bold=true), " ●");
				print(Crayon(foreground=:green, bold=true), "●");
				print(Crayon(foreground=:blue, bold=true), "● ");
				print(Crayon(foreground=:white, bold=true), $expr)
				)
	
	return esc(expr_out)
end

macro boink(expr::Expr)
	"Add EPSILON to a numerical variable or array."

	expr_out = :($expr .+ EPSILON)
	return esc(expr_out)
end

macro positive(expr::Expr)
	"Add EPSILON to a numerical variable or array during variable assignment."

	if (expr.head == :.) || (expr.head == :ref)
		expr_out = :($expr .+= EPSILON)
	
	elseif expr.head == :(=)
		expr_out = :($(expr.args[1]) = EPSILON .+ $(expr.args[2]))
	end

	return esc(expr_out)
end

macro finite(expr::Expr)
	"Prevent overflow of floating point to Inf by returning floatmax() of value."

	if (expr.head == :.) || (expr.head == :ref)
		expr_out = :($expr = sign.($expr) .* min.(abs.($expr), floatmax.($expr)))
	
	elseif expr.head == :(=)
		expr_out = :($(expr.args[1]) = sign.($(expr.args[2])) .* min.(abs.($(expr.args[2])), floatmax.($(expr.args[2]))))

	elseif expr.head == :(-=)
		expr_out = :($(expr.args[1]) = sign.($(expr.args[1])) .* min.(abs.($(expr.args[1]) - $(expr.args[2])), floatmax.($(expr.args[1]) - $(expr.args[2]))))
	end

	return esc(expr_out)
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
		expr_out = :($(esc(model)).sigma = Symmetric(reshape(cl.read($(esc(model)).queue, $(esc(model)).sigma_buffer), $(esc(model)).K, $(esc(model)).K)))

	elseif expr.args[2] == :(:lambda_dist_buffer)
		expr_out = :($(esc(model)).lambda_dist = cl.read($(esc(model)).queue, $(esc(model)).lambda_dist_buffer)[1:$(esc(model)).M])

	elseif expr.args[2] == :(:gimel_buffer)
		expr_out = 
		quote
		gimel_host = reshape(cl.read($(esc(model)).queue, $(esc(model)).gimel_buffer), $(esc(model)).K, $(esc(model)).M)
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
		local kwargs = [(kw.args[1], eval(kw.args[2])) for kw in $(esc(expr.args[3:end]))]
		
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
			gpumodel.phi = [Array{Float32}(undef, 0, 0) for d in 1:model.M]

			for d in 1:model.M
				terms = model.corp[d].terms
				@positive gpumodel.phi[d] = model.beta_old[:,terms] .* exp.(model.Elogtheta_old[d])
				gpumodel.phi[d] ./= sum(gpumodel.phi[d], dims=1)
			end

			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)
			
			model.topics = gpumodel.topics
			model.alpha = gpumodel.alpha
			model.beta = gpumodel.beta
			model.Elogtheta = gpumodel.Elogtheta
			model.Elogtheta_old = deepcopy(model.Elogtheta)
			model.gamma = gpumodel.gamma
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.beta ./= sum(model.beta, dims=2)
			model.beta_old = copy(model.beta)
			model.phi ./= sum(model.phi, dims=1)
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
			gpumodel.sigma = convert(Symmetric{Float32,Array{Float32,2}}, model.sigma)
			gpumodel.invsigma = convert(Symmetric{Float32,Array{Float32,2}}, model.invsigma)
			gpumodel.beta = model.beta
			gpumodel.lambda = model.lambda
			gpumodel.vsq = model.vsq
			gpumodel.logzeta = model.logzeta
			gpumodel.phi = [Array{Float32}(undef, 0, 0) for d in 1:model.M]

			for d in 1:model.M
				terms = model.corp[d].terms
				@positive gpumodel.phi[d] = additive_logistic(log.(model.beta_old[:,terms]) .+ model.lambda_old[d], dims=1)
				gpumodel.phi[d] ./= sum(gpumodel.phi[d], dims=1)
			end

			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)
			
			model.topics = gpumodel.topics
			model.mu = gpumodel.mu
			model.sigma = convert(Symmetric{Float64,Array{Float64,2}}, model.sigma)
			model.invsigma = convert(Symmetric{Float64,Array{Float64,2}}, model.invsigma)
			model.beta = gpumodel.beta
			model.lambda = gpumodel.lambda
			model.lambda_old = deepcopy(model.lambda)
			model.vsq = gpumodel.vsq
			model.logzeta = gpumodel.logzeta
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.beta ./= sum(model.beta, dims=2)
			model.beta_old = copy(model.beta)
			model.phi ./= sum(model.phi, dims=1)
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
			gpumodel.phi = [Array{Float32}(undef, 0, 0) for d in 1:model.M]
			gpumodel.xi = [Array{Float32}(undef, 0, 0) for d in 1:model.M]

			for d in 1:model.M
				terms = model.corp[d].terms
				readers = model.corp[d].readers

				gpumodel.phi[d] = exp.(digamma.(model.gimel_old[d]) - log.(model.dalet_old) - log.(model.bet_old) .+ digamma.(model.alef_old[:,terms]))
				gpumodel.phi[d] ./= sum(gpumodel.phi[d], dims=1)

				gpumodel.xi[d] = vcat(exp.(digamma.(model.gimel_old[d]) - log.(model.dalet_old) - log.(model.vav_old) .+ digamma.(model.he_old[:,readers])), exp.(digamma.(model.zayin_old[d]) - log.(model.het_old) - log.(model.vav_old) .+ digamma.(model.he_old[:,readers])))
				gpumodel.xi[d] ./= sum(gpumodel.xi[d], dims=1)
			end

			gpumodel.elbo = model.elbo
			
			train!(gpumodel; kwargs...)

			model.topics = gpumodel.topics
			model.scores = gpumodel.scores
			model.drecs = gpumodel.drecs
			model.urecs = gpumodel.urecs
			model.alef = gpumodel.alef
			model.alef_old = copy(model.alef)
			model.he = gpumodel.he
			model.he_old = copy(model.he)
			model.bet = gpumodel.bet
			model.bet_old = copy(model.bet)
			model.vav = gpumodel.vav
			model.vav_old = copy(model.vav)
			model.gimel = gpumodel.gimel
			model.gimel_old = deepcopy(model.gimel)
			model.zayin = gpumodel.zayin
			model.zayin_old = deepcopy(model.zayin)
			model.dalet = gpumodel.dalet
			model.dalet_old = copy(model.dalet)
			model.het = gpumodel.het
			model.het_old = copy(model.het)
			model.phi = gpumodel.phi[1:min(gpumodel.M, 1)]
			model.xi = gpumodel.xi[1:min(gpumodel.M, 1)]
			model.elbo = gpumodel.elbo

			model.phi ./= sum(model.phi, dims=1)
			model.xi ./= sum(model.xi, dims=1)
			nothing

		elseif isa(model, fLDA)
			nothing

		elseif isa(model, fCTM)
			nothing

		else
			train!(model; kwargs...)
		end
	end
end
