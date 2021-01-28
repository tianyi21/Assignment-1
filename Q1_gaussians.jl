### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ 19d93518-610d-11eb-3a35-21929592fe91
using Markdown

# ╔═╡ b4c42a46-610e-11eb-1504-a53931294bc7
using Test

# ╔═╡ 83cfb79a-610e-11eb-1b17-abbdcd436b3e
using Distributions: pdf, Normal 
# Note Normal uses N(mean, stddev) for parameters

# ╔═╡ 51dfc82c-610e-11eb-2c6e-c7009592e608
using Statistics: mean, var

# ╔═╡ 66eb5286-610e-11eb-2ec0-534bafa033ca
using Plots

# ╔═╡ 46ec792a-610d-11eb-1dfd-677ecccbdfe6
md"""

# 1D Gaussians [10 pts]

Let $X$ be a univariate random variable distributed according to a Gaussian distribution with mean $\mu$ and variance $\sigma^2$

"""

# ╔═╡ 4f5862be-610c-11eb-3722-4de4ac3c675e
md"""
###  Can the probability density function (pdf) of $X$ ever take values greater than $1$?

Answer:
"""

# ╔═╡ 9da736ec-610d-11eb-004f-c75406dd7ce1
md"""
###  Write the expression for the pdf of a univariate gaussian:

Answer:
"""

# ╔═╡ 763e1346-610d-11eb-305c-7bc070bb0334
md"""
###  Write the code for the function that computes the pdf at $x$.
"""

# ╔═╡ 373706c2-610c-11eb-2273-2be89f7db95e
function gaussian_pdf(x; mean=0., variance=0.01)
  #default variables mean and variance
  #set with keyword arguments
  return #TODO: implement pdf at x
end

# ╔═╡ f2ef3f92-610c-11eb-25d5-cdc66ee3a63a
md"""
### Test your implementation against a standard implementation

E.g. from a library, e.g. Distributions.jl.
"""

# ╔═╡ bbca17dc-610f-11eb-0dc7-43115c69562c


# ╔═╡ 2905a278-610e-11eb-3219-1338acdcdb10
@testset "Implementation of Gaussian pdf" begin
  x = randn()
  @test gaussian_pdf(x) ≈ pdf.(Normal(0.,sqrt(0.01)),x)
  # ≈ is syntax sugar for isapprox, typed with `\approx <TAB>`
  # or use the full function, like below
  @test isapprox(gaussian_pdf(x,mean=10., variance=1) , pdf.(Normal(10., sqrt(1)),x))
end

# ╔═╡ d4f16cb6-610e-11eb-2d58-add315b2ddbd
md"""
###  What is the value of the pdf at $x=0$? What is probability that $x=0$?
"""

# ╔═╡ 79c4c724-610f-11eb-1504-ab62472e6da0


# ╔═╡ d2f2d65c-610e-11eb-15c4-55e479aeb7b0
md"""
###  A Write the transformation that takes $x \sim \mathcal{N}(0.,1.)$ to $z \sim \mathcal{N}(\mu, \sigma^2)$

A Gaussian with mean $\mu$ and variance $\sigma^2$ can be written as a simple transformation of the standard Gaussian with mean $0.$ and variance $1.$.

Answer:
"""

# ╔═╡ 880b0b40-610f-11eb-2e2a-732841b5c779


# ╔═╡ d1739e74-610e-11eb-3f02-d3b44270a23b
md"""
### Write a code to sample from $\mathcal{N}(\mu, \sigma^2)$

Implement function returning $n$ independent samples from $\mathcal{N}(\mu, \sigma^2)$ by transforming $n$ samples from $\mathcal{N}(0.,1.)$
"""

# ╔═╡ ab40ff20-610f-11eb-0563-7ba8a4a6fda9
function sample_gaussian(n; mean=0., variance=0.01)
  # n samples from standard gaussian
  x = #TODO

  # TODO: transform x to sample z from N(mean,variance)
  z = 
  return z
end;

# ╔═╡ a857197a-610f-11eb-30ac-27c2af7f0860
md"""
### Test your implementation by computing statistics on the samples
"""

# ╔═╡ 5222c896-6110-11eb-3e03-3f5ea5d15658
@testset "Numerically testing Gaussian Sample Statistics" begin
  #TODO: choose some values of mean and variance to test
  #TODO: Sample 100000 samples with sample_gaussian
  #TODO: Use `mean` and `var` to compute statistics
  #TODO: test statistics against true values
  # hint: use isapprox with keyword argument atol=1e-2
end;

# ╔═╡ 697c9160-610e-11eb-321a-e3f262d190b4
md"""
### Plot pdf and normalized histogram of samples

Sample $10000$ samples from a Gaussian with mean $10.$ and variance $2.0$. 

1. Plot the **normalized** `histogram` of these samples. 
2. On the same axes `plot!` the pdf of this distribution.
Confirm that the histogram approximates the pdf.

(Note: with `Plots.jl` the function `plot!` will add to the existing axes.)
"""

# ╔═╡ ef08d6be-6110-11eb-031d-bfb21db6f0de
#histogram() #TODO
#plot!() #TODO

# ╔═╡ 5c5c4d7a-610e-11eb-0075-4126b8ba0d23


# ╔═╡ Cell order:
# ╠═19d93518-610d-11eb-3a35-21929592fe91
# ╠═46ec792a-610d-11eb-1dfd-677ecccbdfe6
# ╠═4f5862be-610c-11eb-3722-4de4ac3c675e
# ╟─9da736ec-610d-11eb-004f-c75406dd7ce1
# ╟─763e1346-610d-11eb-305c-7bc070bb0334
# ╠═373706c2-610c-11eb-2273-2be89f7db95e
# ╟─f2ef3f92-610c-11eb-25d5-cdc66ee3a63a
# ╠═bbca17dc-610f-11eb-0dc7-43115c69562c
# ╠═b4c42a46-610e-11eb-1504-a53931294bc7
# ╠═83cfb79a-610e-11eb-1b17-abbdcd436b3e
# ╠═2905a278-610e-11eb-3219-1338acdcdb10
# ╠═d4f16cb6-610e-11eb-2d58-add315b2ddbd
# ╠═79c4c724-610f-11eb-1504-ab62472e6da0
# ╠═d2f2d65c-610e-11eb-15c4-55e479aeb7b0
# ╠═880b0b40-610f-11eb-2e2a-732841b5c779
# ╟─d1739e74-610e-11eb-3f02-d3b44270a23b
# ╠═ab40ff20-610f-11eb-0563-7ba8a4a6fda9
# ╟─a857197a-610f-11eb-30ac-27c2af7f0860
# ╠═51dfc82c-610e-11eb-2c6e-c7009592e608
# ╠═5222c896-6110-11eb-3e03-3f5ea5d15658
# ╠═697c9160-610e-11eb-321a-e3f262d190b4
# ╠═66eb5286-610e-11eb-2ec0-534bafa033ca
# ╠═ef08d6be-6110-11eb-031d-bfb21db6f0de
# ╟─5c5c4d7a-610e-11eb-0075-4126b8ba0d23
