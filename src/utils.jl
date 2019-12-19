### Utilites for TopicModelsVB
### Eric Proffitt
### December 3, 2019

using DelimitedFiles
using Crayons
using SpecialFunctions
using Distributions
using LinearAlgebra
using Random

### The function eps() outputs the machine epsilon of the argument.
### Argument currently set to 1e-14.
### Resulting EPSILON is approx. 1.6e-30.
const EPSILON = eps(1e-14)

### EPSILON32 is 1e-30.
const EPSILON32 = "0.000000000000000000000000000001f"

### Numerical approximation to the digamma function.
### Based on eq. (12), without looking at the accompanying source
### code, of: K. S. Kölbig, "Programs for computing the logarithm of
### the gamma function, and the digamma function, for complex
### argument," Computer Phys. Commun. vol. 4, pp. 221–226 (1972).
const DIGAMMA_cpp =
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

### Algorithm for putting a matrix into reduced row echelon form.
const RREF_cpp =
"""
inline void
rref(long K, long D, global float *A, global float *B)
			
		{
		for (long j=0; j<K; j++)
		{
			float maxval = fabs(A[D + K * j + j]);

			long maxrow = j;

			for (long i=j+1; i<K; i++)
			{
				if (fabs(A[D + K * j + i]) > maxval)
				{
					maxval = fabs(A[D + K * j + i]);
					maxrow = i;
				}
			}
				
			for (long l=0; l<K; l++)
			{
				float tempvarA = A[D + K * l + maxrow];
				float tempvarB = B[D + K * l + maxrow];

				A[D + K * l + maxrow] = A[D + K * l + j];
				B[D + K * l + maxrow] = B[D + K * l + j];

				A[D + K * l + j] = tempvarA;
				B[D + K * l + j] = tempvarB;
			}

			for (long i=j; i<K-1; i++)
			{
				float c = -A[D + K * j + (i + 1)] / A[D + K * j + j];
		
				for (long l=j; l<K; l++)
					if (l == j)
						A[D + K * l + (i + 1)] = 0.0f;
					else
						A[D + K * l + (i + 1)] += c * A[D + K * l + j];
		
				for (long l=0; l<K; l++)	
					B[D + K * l + (i + 1)] += c * B[D + K * l + j];
			}
		}

		for (long j=K-1; j>0; j--)
			for (long i=j-1; i>=0; i--)
				for (long l=0; l<K; l++)
					B[D + K * l + i] -= B[D + K * l + j] * A[D + K * j + i] / A[D + K * j + j];

		for (long j=0; j<K; j++)
			for (long l=0; l<K; l++)
				B[D + K * l + j] *= 1 / A[D + K * j + j];
		}			
		"""

### Algorithm for taking the L2 norm of a vector.
const NORM2_cpp =
"""
inline float
norm2(long K, long d, global float *x)
		
		{
		float acc = 0.0f;

		for (long i=0; i<K; i++)
			acc += x[K * d + i] * x[K * d + i];

		return sqrt(acc);
		}
		"""

"Type alias for a vector of vectors."
VectorList{T} = Vector{Vector{T}}

"Type alias for a vector of matrices."
MatrixList{T} = Vector{Matrix{T}}

function logsumexp(x::Array{<:Real})
	"The LogSumExp of a real-valued array."
	"Overflow safe."

	maxval = maximum(x)
	return maxval + log(sum(exp.(x .- maxval)))
end

function additive_logistic(x::Matrix{<:Real}; dims::Integer)
	"Additive logistic function of a real-valued matrix."
	"Overflow safe."

	@assert dims in [1, 2]

	if dims == 1
		x = x .- [maximum(x[:,j]) for j in 1:size(x, 2)]'
		x = exp.(x) ./ sum(exp.(x), dims=1)

	else
		x = x .- [maximum(x[i,:]) for i in 1:size(x, 1)]
		x = exp.(x) ./ sum(exp.(x), dims=2)
	end

	return x
end

function additive_logistic(x::Vector{<:Real})
	"Additive logistic function of a real-valued vector."
	"Overflow safe."

	x = x .- maximum(x)
	return exp.(x) / sum(exp.(x))
end

function additive_logistic(x::Matrix{<:Real})
	"Additive logistic function of a real-valued Matrix."
	"Overflow safe."

	x = x .- maximum(x)
	return exp.(x) / sum(exp.(x))
end

function Distributions.isprobvec(P::Matrix{<:Real}; dims::Integer)
	"Extend the functionality of the isprobvec function in the Distributions Pkg."
	"Add row and column-wise functionality for isprobvec on real-valued matrices."

	@assert dims in [1, 2]

	if dims == 1
		x = all([isprobvec(P[:,j]) for j in 1:size(P, 2)])

	else
		x = all([isprobvec(P[i,:]) for i in 1:size(P, 1)])
	end

	return x
end