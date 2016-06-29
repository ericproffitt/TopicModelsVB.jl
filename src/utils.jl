typealias VectorList{T} Vector{Vector{T}}
typealias MatrixList{T} Vector{Matrix{T}}

bold(str::AbstractString) = print_with_color(:bold, str)
yellow(str::AbstractString) = print_with_color(:yellow, str)

macro juliadots(stringexpr::Expr)
	stringexpr = :(print_with_color(:red, " ●");
					print_with_color(:green, "●");
					print_with_color(:blue, "● ");
					print_with_color(:bold, $stringexpr))
	return stringexpr
end

isnegative(x::Real) = x < 0
ispositive(x::Real) = x > 0
isnegative{T<:Real}(xs::Array{T}) = Bool[isnegative(x) for x in xs]
ispositive{T<:Real}(xs::Array{T}) = Bool[ispositive(x) for x in xs]
tetragamma(x) = polygamma(2, x)

Distributions.isprobvec(P::Matrix{Float64}) = isprobvec(vcat(P...))

function Distributions.isprobvec(P::Matrix{Float64}, region::Int)
	@assert (isequal(region, 1) | isequal(region, 2))

	if region == 1
		x = all([isprobvec(P[:,j]) for j in 1:size(P, 2)])
	else
		x = all([isprobvec(vec(P[i,:])) for i in 1:size(P, 1)])
	end
	return x
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

