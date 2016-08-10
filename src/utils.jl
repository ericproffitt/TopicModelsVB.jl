const epsln = eps(1e-14)

typealias VectorList{T} Vector{Vector{T}}
typealias MatrixList{T} Vector{Matrix{T}}

bold(str::AbstractString) = print_with_color(:bold, str)
yellow(str::AbstractString) = print_with_color(:yellow, str)

isnegative(x::Real) = x < 0
ispositive(x::Real) = x > 0
isnegative{T<:Real}(xs::Array{T}) = Bool[isnegative(x) for x in xs]
ispositive{T<:Real}(xs::Array{T}) = Bool[ispositive(x) for x in xs]

function logsumexp{T<:Real}(xs::Array{T})
	maxval = maximum(xs)
	return maxval + log(sum(exp(xs - maxval)))
end

function addlogistic{T<:Real}(xs::Array{T})
	maxval = maximum(xs)
	xs -= maxval
	xs = exp(xs) / sum(exp(xs))
	return xs
end

function addlogistic{T<:Real}(xs::Matrix{T}, region::Int)
	if region == 1
		maxvals = [maximum(xs[:,j]) for j in 1:size(xs, 2)]
		xs .-= maxvals'
		xs = exp(xs) ./ sum(exp(xs), 1)
	elseif region == 2
		maxvals = [maximum(xs[i,:]) for i in 1:size(xs, 1)]
		xs .-= maxvals
		xs = exp(xs) ./ sum(exp(xs), 2)
	else
		xs = addlogistic(xs)
	end
	return xs
end

Distributions.isprobvec(p::Vector{Float32}) = isapprox(sum(p), 1.0f0)
Distributions.isprobvec{T<:Real}(P::Matrix{T}) = isprobvec(vcat(P...))

function Distributions.isprobvec{T<:Real}(P::Matrix{T}, region::Int)
	@assert (isequal(region, 1) | isequal(region, 2))

	if region == 1
		x = all([isprobvec(P[:,j]) for j in 1:size(P, 2)])
	elseif region == 2
		x = all([isprobvec(vec(P[i,:])) for i in 1:size(P, 1)])
	end
	return x
end

function Distributions.Categorical(p::Vector{Float32})
	@assert isapprox(sum(p), 1)
	p = map(Float64, p)
	p /= sum(p)
	return Categorical(p)
end

function Distributions.Multinomial(n::Integer, p::Vector{Float32})
	@assert isapprox(sum(p), 1)
	p = map(Float64, p)
	p /= sum(p)
	return Multinomial(n, p)
end

function partition{T<:Any}(xs::Vector{T}, n::Int)
	@assert ispositive(n)

	q = div(length(xs), n)
	r = length(xs) - q*n
	p = Vector{T}[xs[(n*i-n+1):(n*i)] for i in 1:q]
	if ispositive(r)
		push!(p, xs[(q*n+1):end])
	end
	return p
end

partition{T<:Real}(xs::UnitRange{T}, n::Int) = partition(collect(xs), n)

const epsln32 = "0.000000000000000000000000000001f"

const digammacpp =
"""
inline float
digamma(float x)
		
	{
	float p = 0.0f;

	if (x < 7)
	{
		int n = 7 - floor(x);		
		for (int v=1; v < n; v++)
			p -= 1 / (x + v);
	        
		p -= 1 / x;
		x += n;
	}
	    
	float t = 1 / x;
	p += log(x) - 0.5f * t;
	t *= t;
	p -= t * 0.08333333333333333f
			- 0.008333333333333333f * t
			+ 0.003968253968253968f * t*t
			- 0.004166666666666667f * t*t*t
			+ 0.007575757575757576f * t*t*t*t
			- 0.021092796092796094f * t*t*t*t*t
			+ 0.08333333333333333f * t*t*t*t*t*t
			- 0.4432598039215686f * t*t*t*t*t*t*t;
	return p;
	}
	"""

const trigammacpp =
"""
"""
