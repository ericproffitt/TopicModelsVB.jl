## The function eps() outputs the machine epsilon of the argument.
## eps(1e-14) ≈ 1.6e-30.
const EPSILON = eps(1e-14)

## EPSILON32 is 1f-30.
const EPSILON32 = "0.000000000000000000000000000001f"

## Euler–Mascheroni constant.
## γ = 0.5772156649015...
import Base.MathConstants.eulergamma

#=
Numerical approximation to the digamma function.
Based on eq. (12), without looking at the accompanying source
code, of: K. S. Kölbig, "Programs for computing the logarithm of
the gamma function, and the digamma function, for complex
argument," Computer Phys. Commun. vol. 4, pp. 221–226 (1972).
=#
const DIGAMMA_c =
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

#=
Parallel Gauss-Jordan elimination (reduced row echelon form) algorithm for
solving the linear system Ax=b. Partial pivoting unnecessary for symmetric
positive-definite matrices. Ref, "Accuracy and Stability of Numerical Algorithms" Higham, 2002.
=#
const LINSOLVE_c =
"""
inline void
linsolve(long K, long z, local float *A, local float *b)
			
		{
		for (long j=0; j<K-1; j++)
		{
			if ((j <= z) && (z < K-1))
			{
				float c = -A[K * j + (z + 1)] / A[K * j + j];

			    for (long l=j+1; l<K; l++)
			        A[K * l + (z + 1)] += c * A[K * l + j];

			    b[z + 1] += c * b[j];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		for (long j=K-1; j>0; j--)
		{
			if (z < j)
				b[z] -= b[j] * A[K * j + z] / A[K * j + j];

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		b[z] *= 1 / A[K * z + z];
		}			
		"""

"Type alias for a vector of vectors."
VectorList{T} = Vector{Vector{T}}

"Type alias for a vector of matrices."
MatrixList{T} = Vector{Matrix{T}}

"Prevent overflow to ±Inf."
finite(x::Union{AbstractFloat, Array{<:AbstractFloat}}) = sign.(x) .* min.(abs.(x), floatmax.(x))

## LogSumExp function, overflow safe.
import Distributions.logsumexp

function additive_logistic(x::Matrix{<:Real}; dims::Integer)
	"""
	Additive logistic function of a real-valued matrix over the given dimension.
	Overflow safe.
	"""

	if dims in [1,2]
		x = exp.(x .- maximum(x, dims=dims))
		x = x ./ sum(x, dims=dims)
	end

	return x
end

function additive_logistic(x::Vector{<:Real})
	"""
	Additive logistic function of a real-valued vector.
	Overflow safe.
	"""

	x = exp.(x .- maximum(x))
	x = x / sum(x)

	return x
end

function additive_logistic(x::Matrix{<:Real})
	"""
	Additive logistic function of a real-valued matrix.
	Overflow safe.
	"""

	x = exp.(x .- maximum(x))
	x = x / sum(x)

	return x
end

function isstochastic(P::Matrix{<:Real}; dims::Integer)
	"""
	Check to see if X is a stochastic matrix.
		if dims = 1, check for left stochastic matrix.
		if dims = 2, check for right stochastic matrix.
	"""

	if dims == 1
		x = all([isprobvec(P[:,j]) for j in 1:size(P, 2)])
	elseif dims == 2
		x = all([isprobvec(P[i,:]) for i in 1:size(P, 1)])
	else
		x = P
	end

	return x
end

## Keep until pull request is merged.
import Distributions.xlogy
import Distributions.binomlogpdf
Distributions.xlogy(x::T, y::T) where {T<:Real} = x != zero(T) ? x * log(y) : zero(log(y))
Distributions.binomlogpdf(n::Real, p::Real, k::Real) = (isinteger(k) & (zero(k) <= k <= n)) ? convert(typeof(float(p)), loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1) + xlogy(k, p) + xlogy(n - k, 1 - p)) : convert(typeof(float(p)), -Inf)

## Keep until pull request is merged.
function Distributions.entropy(d::Dirichlet)
    α = d.alpha
    α0 = d.alpha0
    k = length(α)

    if k == 1
    	en = 0.0

    else
	    en = d.lmnB + (α0 - k) * digamma(α0)
	    for j in 1:k
	        @inbounds αj = α[j]
	        en -= (αj - 1.0) * digamma(αj)
	    end
	end

    return en
end
