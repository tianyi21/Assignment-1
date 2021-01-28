### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ aa84f014-6111-11eb-25e5-c7bec21824e9
md"""
# High-Dimensional Gaussians [20 pts]

In this question we will investigate how our intuition for samples from a Gaussian may break down in higher dimensions. Consider samples from a $D$-dimensional unit Gaussian

  $x \sim \mathcal{N}(0_D, I_D)$
where~$0_D$ indicates a column vector of~$D$ zeros and~$I_D$
is a ${D\times D}$ identity matrix.
"""


# ╔═╡ 0634fd58-6112-11eb-37f6-45112ee734ae
md"""
### Distance of Gaussian samples from origin

Starting with the definition of Euclidean norm, quickly show that the distance of $x$ from the origin is $\sqrt{ x ^ \intercal x }$
"""

# ╔═╡ 08b6ed2a-6112-11eb-3277-69f7c404be51
md"""
### Distribution of distances of Gaussian samples from origin

In low-dimensions our intuition tells us that samples from the unit Gaussian will be near the origin. 

1. Draw 10000 samples from a $D=1$ Gaussian
2. Compute the distance of those samples from the origin.
3. Plot a normalized histogram for the distance of those samples from the origin. 

Does this confirm your intuition that the samples will be near the origin?
"""

# ╔═╡ 86371c3e-6112-11eb-1660-f32994a6b1a5


# ╔═╡ 117c783a-6112-11eb-0cfc-bb24a3234baf
"""md
### Draw 10000 samples from $D=\{1,2,3,10,100\}$ Gaussians and, on a single plot, show the normalized histograms for the distance of those samples from the origin. As the dimensionality of the Gaussian increases, what can you say about the expected distance of the samples from the Gaussian's mean (in this case, origin).

Answer
"""

# ╔═╡ 1ccebb58-6112-11eb-3028-cff830e3a9e8
md"""
### Plot samples from distribution of distances 

1. Draw a set of 10000 samples from $D=\{1,2,3,10,100\}$ Gaussians
2. Compute the distance of each sample from the origin
2. With all D dimensionality on a single plot, show the normalized histograms for the distribution of distance of those samples from the origin. 

As the dimensionality of the Gaussian increases, what can you say about the expected distance of the samples from the Gaussian's mean (in this case, origin)?
"""

# ╔═╡ 387dc1de-6174-11eb-069c-e70e4483ea67
md"""
### Plot the $\chi$-distribution

From Wikipedia, if $x_i$ are $k$ independent, normally distributed random variables
with means $\mu_i$ and standard deviations $\sigma_i$ then the statistic $Y =
\sqrt{\sum_{i=1}^k(\frac{x_i-\mu_i}{\sigma_i})^2}$ is distributed according to the
[$\chi$-distribution](https://en.wikipedia.org/wiki/Chi_distribution)] 

On the previous normalized histogram, plot the probability density function (pdf) 
of the $\chi$-distribution for $k=\{1,2,3,10,100\}$.
"""

# ╔═╡ 67cc8c54-6174-11eb-02d1-95d31e908329


# ╔═╡ 6865c7f2-6174-11eb-2d0b-c73f00d5c347
md"""
### Distribution of distance between samples

Taking two samples from the $D$-dimensional unit Gaussian,
$x_a, x_b \sim \mathcal{N}(  0_D, I_D)$ how is $x_a -  x_b$ distributed?
Using the above result about $\chi$-distribution, derive how $\vert \vert x _a -  x _b\vert \vert_2$ is distributed.


(Hint: start with a $\mathcal{X}$-distributed random variable and use the [change of variables formula](https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables).)
"""

# ╔═╡ dbb1e2c2-6174-11eb-0b7b-7b93b3e3444c


# ╔═╡ dc4a3644-6174-11eb-3e97-0143edb860f8
md"""
### Plot pdfs of distribution distances between samples 

For for $D=\{1,2,3,10,100\}$. 
    How does the distance between samples from a Gaussian behave as dimensionality increases?
    Confirm this by drawing two sets of $1000$ samples from the $D$-dimensional unit Gaussian.
    On the plot of the $\chi$-distribution pdfs, plot the normalized histogram of the distance between samples from the first and second set.
"""

# ╔═╡ 171207b6-6175-11eb-2467-cdfb7e1fd324


# ╔═╡ 18adf0b2-6175-11eb-1753-a7f33f0d7ca3
md"""
### Linear interpolation between samples

Given two samples from a gaussian $x_a,x_b \sim \mathcal{N}(  0 _D, I_D)$ the
    linear interpolation between them $x_\alpha$ is defined as a function of  $\alpha\in[0,1]$

$$\text{lin\_interp}(\alpha, x _a, x _b) = \alpha  x _a + (1-\alpha) x _b$$

For two sets of 1000 samples from the unit gaussian in $D$-dimensions, plot the average log-likelihood along the linear interpolations between the pairs of samples as a function of $\alpha$.

(i.e. for each pair of samples compute the log-likelihood along a linear space of interpolated points between them, $\mathcal{N}(x_\alpha|0,I)$ for $\alpha \in [0,1]$. Plot the average log-likelihood over all the interpolations.)


Do this for $D=\{1,2,3,10,100\}$, one plot per dimensionality.
Comment on the log-likelihood under the unit Gaussian of points along the linear interpolation.
Is a higher log-likelihood for the interpolated points necessarily better?
Given this, is it a good idea to linearly interpolate between samples from a high dimensional Gaussian?
"""

# ╔═╡ a738e7ba-6175-11eb-0103-fb6319b44ece
md"""
###  Polar Interpolation Between Samples

Instead we can interpolate in polar coordinates: For $\alpha\in[0,1]$ the polar interpolation is

$$\text{polar\_interp}(\alpha, x _a, x _b)=\sqrt{\alpha } x _a + \sqrt{(1-\alpha)} x _b$$

This interpolates between two points while maintaining Euclidean norm.


On the same plot from the previous question, plot the probability density of the polar interpolation between pairs of samples from two sets of 1000 samples from $D$-dimensional unit Gaussians for $D=\{1,2,3,10,100\}$. 

Comment on the log-likelihood under the unit Gaussian of points along the polar interpolation.
Give an intuitive explanation for why polar interpolation is more suitable than linear interpolation for high dimensional Gaussians. 
(For 6. and 7. you should have one plot for each $D$ with two curves on each).
"""

# ╔═╡ d0b81a0c-6175-11eb-3005-811ab72f7077


# ╔═╡ e3b3cd7c-6111-11eb-093e-7ffa8410b742
md"""

### Norm along interpolation

In the previous two questions we compute the average log-likelihood of the linear and polar interpolations under the unit gaussian.
Instead, consider the norm along the interpolation, $\sqrt{ x _\alpha^ \intercal x _\alpha}$.
As we saw previously, this is distributed according to the $\mathcal{X}$-distribution.
Compute and plot the average log-likelihood of the norm along the two interpolations under the the $\mathcal{X}$-distribution for $D=\{1,2,3,10,100\}$, 
i.e. $\mathcal{X}_D(\sqrt{ x _\alpha^ \intercal x _\alpha})$. 
There should be one plot for each $D$, each with two curves corresponding to log-likelihood of linear and polar interpolations.
How does the log-likelihood along the linear interpolation compare to the log-likelihood of the true samples (endpoints)?
"""


# ╔═╡ 07176c60-6176-11eb-0336-db9f450ed67f


# ╔═╡ Cell order:
# ╟─aa84f014-6111-11eb-25e5-c7bec21824e9
# ╟─0634fd58-6112-11eb-37f6-45112ee734ae
# ╟─08b6ed2a-6112-11eb-3277-69f7c404be51
# ╠═86371c3e-6112-11eb-1660-f32994a6b1a5
# ╟─117c783a-6112-11eb-0cfc-bb24a3234baf
# ╟─1ccebb58-6112-11eb-3028-cff830e3a9e8
# ╟─387dc1de-6174-11eb-069c-e70e4483ea67
# ╠═67cc8c54-6174-11eb-02d1-95d31e908329
# ╟─6865c7f2-6174-11eb-2d0b-c73f00d5c347
# ╠═dbb1e2c2-6174-11eb-0b7b-7b93b3e3444c
# ╟─dc4a3644-6174-11eb-3e97-0143edb860f8
# ╠═171207b6-6175-11eb-2467-cdfb7e1fd324
# ╟─18adf0b2-6175-11eb-1753-a7f33f0d7ca3
# ╟─a738e7ba-6175-11eb-0103-fb6319b44ece
# ╠═d0b81a0c-6175-11eb-3005-811ab72f7077
# ╟─e3b3cd7c-6111-11eb-093e-7ffa8410b742
# ╠═07176c60-6176-11eb-0336-db9f450ed67f
