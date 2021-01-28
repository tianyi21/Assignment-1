### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ 2038c102-6177-11eb-047e-95829eaf59b7
using LinearAlgebra

# ╔═╡ 8625e7ba-6177-11eb-0f73-f14acfe6a2ea
using Test

# ╔═╡ c1c7bd02-6177-11eb-3073-d92dde78c4bb
using Plots

# ╔═╡ d2cb6eda-6179-11eb-3e11-41b238bd071d
using Zygote: gradient

# ╔═╡ 3b614428-6176-11eb-27af-1d611c78a404
md"""
# Regression [30pts]
"""

# ╔═╡ 4c8b7872-6176-11eb-3c4c-117dfe425f8a
md"""
### Manually Derived Linear Regression

Suppose that 
$X \in \mathbb{R}^{m \times n}$ with $n \geq m$ 
and $Y \in \mathbb{R}^n$, and that $Y \sim \mathcal{N}(X^T\beta, \sigma^2 I)$.

In this question you will derive the result that the maximum likelihood estimate $\hat\beta$ of $\beta$ is given by

$$\hat\beta = (XX^T)^{-1}XY$$

1. What happens if $n < m$?

2. What are the expectation and covariance matrix of $\hat\beta$, for a given true value of $\beta$?

3. Show that maximizing the likelihood is equivalent to minimizing the squared error $\sum_{i=1}^n (y_i - x_i\beta)^2$. [Hint: Use $\sum_{i=1}^n a_i^2 = a^Ta$]

4. Write the squared error in vector notation, (see above hint), expand the expression, and collect like terms. [Hint: Use $\beta^Tx^Ty = y^Tx\beta$ and $x^Tx$ is symmetric]

5. Use the likelihood expression to write the negative log-likelihood.
    Write the derivative of the negative log-likelihood with respect to $\beta$, set equal to zero, and solve to show the maximum likelihood estimate $\hat\beta$ as above. 
"""

# ╔═╡ 16c9fa00-6177-11eb-27b4-175a938132af
md"""
### Toy Data

For visualization purposes and to minimize computational resources we will work with 1-dimensional toy data. 

That is $X \in \mathbb{R}^{m \times n}$ where $m=1$.

We will learn models for 3 target functions

* `target_f1`, linear trend with constant noise.
* `target_f2`, linear trend with heteroskedastic noise.
* `target_f3`, non-linear trend with heteroskedastic noise.
"""

# ╔═╡ 361f0788-6177-11eb-35a9-d12176832720
function target_f1(x, σ_true=0.3)
  noise = randn(size(x))
  y = 2x .+ σ_true.*noise
  return vec(y)
end

# ╔═╡ 361f450e-6177-11eb-1320-9b16d3cbe14c
function target_f2(x)
  noise = randn(size(x))
  y = 2x + norm.(x)*0.3.*noise
  return vec(y)
end

# ╔═╡ 361f9f40-6177-11eb-0792-3f7214a6dafe
function target_f3(x)
  noise = randn(size(x))
  y = 2x + 5sin.(0.5*x) + norm.(x)*0.3.*noise
  return vec(y)
end

# ╔═╡ 61d33d5e-6177-11eb-17be-ff5d831900fa
md"""
### Sample data from the target functions 

Write a function which produces a batch of data $x \sim \text{Uniform}(0,20)$ and `y = target_f(x)`
"""

# ╔═╡ 6ac7bb10-6177-11eb-02bc-3513af63a9b9
md""" 
### Test assumptions about your dimensions
"""

# ╔═╡ 91b725a8-6177-11eb-133f-cb075b2c50dc
md"""
### Plot the target functions

For all three targets, plot a $n=1000$ sample of the data.

**Note: You will use these plots later, in your writeup.
Conmsider suppressing display once other questions are complete.**
"""

# ╔═╡ 2be4e23c-6178-11eb-3019-27d3ac7e6d2e
x1,y1  #TODO

# ╔═╡ 2be53474-6178-11eb-2b75-1f1c3187770f
plot_f1  #TODO

# ╔═╡ 2be5b9b4-6178-11eb-3e01-61fea97cdfa6
x2,y2 #TODO

# ╔═╡ 2bf298d2-6178-11eb-23e3-77a2347ed0a2
plot_f2 #TODO

# ╔═╡ 2bf9191e-6178-11eb-2db3-f358d690b2a3
x3,y3  #TODO

# ╔═╡ 2bffc028-6178-11eb-0718-199b410131eb
plot_f3  #TODO

# ╔═╡ f6999bb8-6177-11eb-3a0e-35e7a718e0ab
md"""## Linear Regression Model with $\hat \beta$ MLE"""

# ╔═╡ f9f79404-6177-11eb-2d31-857cb99cf7db
md"""
### Code the hand-derived MLE

Program the function that computes the the maximum likelihood estimate given $X$ and $Y$.
    Use it to compute the estimate $\hat \beta$ for a $n=1000$ sample from each target function.
"""

# ╔═╡ 1cfe23f0-6178-11eb-21bf-8d8a46b0d7c6
function beta_mle(X,Y)
  beta = #TODO
  return beta
end

# ╔═╡ 241fda48-6178-11eb-1775-232a4b55d258
x_1, y_1  #TODO

# ╔═╡ 242049d0-6178-11eb-2572-ad3f6ea43127
β_mle_1  #TODO

# ╔═╡ 242af220-6178-11eb-3bfe-6bcf3e814ff1
x_2, y_2  #TODO

# ╔═╡ 24329fca-6178-11eb-0bdc-45e28e499635
β_mle_2  #TODO

# ╔═╡ 24330276-6178-11eb-0a8f-0f49cb7d57fa
x_3, y_3 #TODO

# ╔═╡ 243e38ee-6178-11eb-1c09-71018451b8f7
β_mle_3  #TODO

# ╔═╡ 4f04aca2-6178-11eb-0e59-df88659f11c1
md"""
### Plot the MLE linear regression model

For each function, plot the linear regression model given by $Y \sim \mathcal{N}(X^T\hat\beta, \sigma^2 I)$ for $\sigma=1.$.
    This plot should have the line of best fit given by the maximum likelihood estimate, as well as a shaded region around the line corresponding to plus/minus one standard deviation (i.e. the fixed uncertainty $\sigma=1.0$).
    Using `Plots.jl` this shaded uncertainty region can be achieved with the `ribbon` keyword argument.
    **Display 3 plots, one for each target function, showing samples of data and maximum likelihood estimate linear regression model**
"""

# ╔═╡ 6222e9e0-6178-11eb-3448-57e951600ef9
plot!(plot_f1,TODO)

# ╔═╡ 705a7d28-6178-11eb-280d-315614ad9080
plot!(plot_f2,TODO)

# ╔═╡ 742afde4-6178-11eb-08b6-e77847b0aa40
plot!(plot_f3,TODO)

# ╔═╡ 7f61ff44-6178-11eb-040b-2f19e52ebaf5
md"""
## Log-likelihood of Data Under Model

### Code for Gaussian Log-Likelihood

Write code for the function that computes the likelihood of $x$ under the Gaussian distribution $\mathcal{N}(μ,σ)$.
For reasons that will be clear later, this function should be able to broadcast to the case where $x, \mu, \sigma$ are all vector valued
and return a vector of likelihoods with equivalent length, i.e., $x_i \sim \mathcal{N}(\mu_i,\sigma_i)$.
"""

# ╔═╡ baf4cbea-6178-11eb-247f-4961c91da925
function gaussian_log_likelihood(μ, σ, x)
  """
  compute log-likelihood of x under N(μ,σ)
  """
  return #TODO: log-likelihood function
end

# ╔═╡ ccedbd3e-6178-11eb-2d26-01016c9dea4b
md"""
### Test Gaussian likelihood against standard implementation
"""

# ╔═╡ 045fd22a-6179-11eb-3901-8b327ed3b7a0
md"""
### Model Negative Log-Likelihood

Use your gaussian log-likelihood function to write the code which computes the negative log-likelihood of the target value $Y$ under the model $Y \sim \mathcal{N}(X^T\beta, \sigma^2*I)$ for 
a given value of $\beta$.
"""

# ╔═╡ 2b7be59c-6179-11eb-3fde-01eac8b60a9d
function lr_model_nll(β,x,y;σ=1.)
  return #TODO: Negative Log Likelihood
end

# ╔═╡ 314fd0fa-6179-11eb-0d0c-a7efdfa33c65
md"""
### Compute Negative-Log-Likelihood on data

Use this function to compute and report the negative-log-likelihood of a $n\in \{10,100,1000\}$ batch of data
under the model with the maximum-likelihood estimate $\hat\beta$ and $\sigma \in \{0.1,0.3,1.,2.\}$ for each target function.
"""

# ╔═╡ 5b1dec3c-6179-11eb-2968-d361e1c18cc8
for n in (10,100,1000)
    println("--------  $n  ------------")
    for target_f in (target_f1,target_f2, target_f3)
      println("--------  $target_f  ------------")
      for σ_model in (0.1,0.3,1.,2.)
        println("--------  $σ_model  ------------")
        x,y = #TODO
        β_mle = #TODO
        nll = #TODO
        println("Negative Log-Likelihood: $nll")
      end
    end
end

# ╔═╡ 6cdd6d12-6179-11eb-1115-cd43e97e9a60
md"""
### Effect of model variance

For each target function, what is the best choice of $\sigma$? 


Please note that $\sigma$ and batch-size $n$ are modelling hyperparameters. 
In the expression of maximum likelihood estimate, $\sigma$ or $n$ do not appear, and in principle shouldn't affect the final answer.
However, in practice these can have significant effect on the numerical stability of the model. 
Too small values of $\sigma$ will make data away from the mean very unlikely, which can cause issues with precision.
Also, the negative log-likelihood objective involves a sum over the log-likelihoods of each datapoint. This means that with a larger batch-size $n$, there are more datapoints to sum over, so a larger negative log-likelihood is not necessarily worse. 
The take-home is that you cannot directly compare the negative log-likelihoods achieved by these models with different hyperparameter settings.
"""

# ╔═╡ 97606f8a-6179-11eb-3dbb-af72ec35e4a1
md"""
## Automatic Differentiation and Maximizing Likelihood

In a previous question you derived the expression for the derivative of the negative log-likelihood with respect to $\beta$.
We will use that to test the gradients produced by automatic differentiation.
"""

# ╔═╡ abe7efdc-6179-11eb-14ed-a3b0462cc2f0
md"""
### Compute Gradients with AD, Test against hand-derived
For a random value of $\beta$, $\sigma$, and $n=100$ sample from a target function,
    use automatic differentiation to compute the derivative of the negative log-likelihood of the sampled data
with respect $\beta$.
Test that this is equivalent to the hand-derived value.
"""

# ╔═╡ d9d4d6da-6179-11eb-0165-95d79e1ab92d
md"""
### Train Linear Regression Model with Gradient Descent

In this question we will compute gradients of of negative log-likelihood with respect to $\beta$.
We will use gradient descent to find $\beta$ that maximizes the likelihood.

Write a function `train_lin_reg` that accepts a target function and an initial estimate for $\beta$ and some 
hyperparameters for batch-size, model variance, learning rate, and number of iterations.

Then, for each iteration:
* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\beta$
* update the estimate of $\beta$ with gradient descent with specified learning rate
and, after all iterations, returns the final estimate of $\beta$.
"""

# ╔═╡ 033afc0c-617a-11eb-32a9-f3f467476a0a
using Logging # Print training progress to REPL, not pdf

function train_lin_reg(target_f, β_init; bs= 100, lr = 1e-6, iters=1000, σ_model = 1.)
    β_curr = β_init
    for i in 1:iters
      x,y = #TODO
      @info "loss: $()  β: $β_curr" #TODO: log loss, if you want to monitor training progress
      grad_β = #TODO: compute gradients
      β_curr = #TODO: gradient descent
    end
    return β_curr
end

# ╔═╡ 217b1466-617a-11eb-19c6-2f29ef5d3576
md"""
### Parameter estimate by gradient descent

For each target function, start with an initial parameter $\beta$, 
    learn an estimate for $\beta_\text{learned}$ by gradient descent.
    Then plot a $n=1000$ sample of the data and the learned linear regression model with shaded region for uncertainty corresponding to plus/minus one standard deviation.
"""

# ╔═╡ 449d93e6-617a-11eb-3289-a931618f4bba
β_init = 1000*randn() # Initial parameter

# ╔═╡ 4adae2ac-617a-11eb-0ed7-cdb1985fad44
β_learned  #TODO: call training function

# ╔═╡ 4d628d18-617a-11eb-1944-c922f564a1f3
md"""
### Plot learned models

For each target function, start with an initial parameter $\beta$, 
learn an estimate for $\beta_\text{learned}$ by gradient descent.
Then plot a $n=1000$ sample of the data and the learned linear regression model with shaded region for uncertainty corresponding to plus/minus one standard deviation.
"""

# ╔═╡ 4adb7980-617a-11eb-041d-214170284e3a
#TODO: For each target function, plot data samples and learned regression

# ╔═╡ 78cc51f0-617a-11eb-3408-83b8d44df832
md"""
Non-linear Regression with a Neural Network

In the previous questions we have considered a linear regression model 

$$Y \sim \mathcal{N}(X^T \beta, \sigma^2)$$

This model specified the mean of the predictive distribution for each datapoint by the product of that datapoint with our parameter.

Now, let us generalize this to consider a model where the mean of the predictive distribution is a non-linear function of each datapoint.
We will have our non-linear model be a simple function called `neural_net` with parameters $\theta$ 
(collection of weights and biases).

$$Y \sim \mathcal{N}(\texttt{neural\_net}(X,\theta), \sigma^2)$$
"""

# ╔═╡ 8d762018-617a-11eb-3def-27d3959bb155
md"""
### Fully-connected Neural Network

Write the code for a fully-connected neural network (multi-layer perceptron) with one 10-dimensional hidden layer and a `tanh` nonlinearirty.
You must write this yourself using only basic operations like matrix multiply and `tanh`, you may not use layers provided by a library.

This network will output the mean vector, test that it outputs the correct shape for some random parameters.
"""

# ╔═╡ a95e69f2-617a-11eb-3034-e12b72492357
# Neural Network Function
function neural_net(x,θ)
  return #TODO
end

# ╔═╡ bfa8826a-617a-11eb-2be8-a3fe151a32cb
# Random initial Parameters
θ #TODO

# ╔═╡ ca10a7fa-617a-11eb-1568-4724c3686b01
md"""
### Test assumptions about model output

Test, at least, the dimension assumptions.
"""

# ╔═╡ e3a14bac-617a-11eb-2155-ff812544df13
md"""
### Negative Log-likelihood of NN model
Write the code that computes the negative log-likelihood for this model where the mean is given by the output of the neural network and $\sigma = 1.0$
"""

# ╔═╡ f65af5b8-617a-11eb-3865-7fa972a6a821
function nn_model_nll(θ,x,y;σ=1)
  return #TODO
end

# ╔═╡ 0f355c54-617b-11eb-2768-7b8066538440
md"""
### Training model to maximize likelihood

Write a function `train_nn_reg` that accepts a target function and an initial estimate for $\theta$ and some 
    hyperparameters for batch-size, model variance, learning rate, and number of iterations.

Then, for each iteration:
* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\theta$
* update the estimate of $\theta$ with gradient descent with specified learning rate
and, after all iterations, returns the final estimate of $\theta$.
"""

# ╔═╡ 3ca4bbbc-617b-11eb-36e4-ad24b86bfd19
function train_nn_reg(target_f, θ_init; bs= 100, lr = 1e-5, iters=1000, σ_model = 1. )
    θ_curr = θ_init
    for i in 1:iters
      x,y = #TODO
      @info "loss: $()" #TODO: log loss, if you want to montior training
      grad_θ = #TODO: compute gradients
      θ_curr = #TODO: gradient descent
    end
    return θ_curr
end

# ╔═╡ 429d9e76-617b-11eb-161c-2b16653d2b0c
md"""
### Learn model parameters

For each target function, start with an initialization of the network parameters, $\theta$,
    use your train function to minimize the negative log-likelihood and find an estimate for $\theta_\text{learned}$ by gradient descent.
    
"""

# ╔═╡ 51a22d88-617b-11eb-27cd-158ba4b38acc
θ_init = #TODO
θ_learned = #TODO

# ╔═╡ 5b9a5c98-617b-11eb-2ede-9dbbeb8ea32d
md"""
### Plot neural network regression

Then plot a $n=1000$ sample of the data and the learned regression model with shaded uncertainty bounds given by $\sigma = 1.0$
"""

# ╔═╡ 89cdc082-617b-11eb-3491-453302b03caa
md"""
## Input-dependent Variance

In the previous questions we've gone from a gaussian model with mean given by linear combination

$$Y \sim \mathcal{N}(X^T \beta, \sigma^2)$$

to gaussian model with mean given by non-linear function of the data (neural network)

$$Y \sim \mathcal{N}(\texttt{neural\_net}(X,\theta), \sigma^2)$$

However, in all cases we have considered so far, we specify a fixed variance for our model distribution.
We know that two of our target datasets have heteroscedastic noise, meaning any fixed choice of variance will poorly model the data.

In this question we will use a neural network to learn both the mean and log-variance of our gaussian model.

$$\begin{align*}
\mu, \log \sigma &= \texttt{neural\_net}(X,\theta)\\
Y &\sim \mathcal{N}(\mu, \exp(\log \sigma)^2)
\end{align*}$$
"""

# ╔═╡ b04e11d0-617b-11eb-35a4-b5aeeafb8570
md"""
### Neural Network that outputs log-variance

Write the code for a fully-connected neural network (multi-layer perceptron) with one 10-dimensional hidden layer and a `tanh` nonlinearirty, and outputs both a vector for mean and $\log \sigma$. 
"""

# ╔═╡ c7ae64a6-617b-11eb-10b8-712abc9dc2a6
# Neural Network with variance
function neural_net_w_var(x,θ)
  return #TODO
end

# ╔═╡ d6ad1a06-617b-11eb-224c-0307a6e6f80a
# Random initial Parameters
θ  #TODO

# ╔═╡ c82cff46-617b-11eb-108a-a9576c728328
md"""
### Test model assumptions

Test the output shape is as expected.
"""

# ╔═╡ 648df4f8-6177-11eb-225e-8f9955713b67
function sample_batch(target_f, batch_size)
  x =#TODO
  y =#TODO
  return (x,y)
end

# ╔═╡ e7cddc4e-617b-11eb-3b2c-f7c06fc61fd5
md"""
### Negative log-likelihood with modelled variance

Write the code that computes the negative log-likelihood for this model where the mean and $\log \sigma$ is given by the output of the neural network.
    
(Hint: Don't forget to take $\exp \log \sigma$)
"""

# ╔═╡ 0b7d0ac0-617c-11eb-24c5-b3126ee28f5a
function nn_with_var_model_nll(θ,x,y)
  return #TODO
end

# ╔═╡ 128daf4a-617c-11eb-3c62-1b61708169e0
md"""
### Write training loop

Write a function `train_nn_w_var_reg` that accepts a target function and an initial estimate for $\theta$ and some 
    hyperparameters for batch-size, learning rate, and number of iterations.

Then, for each iteration:

* sample data from the target function
* compute gradients of negative log-likelihood with respect to $\theta$
* update the estimate of $\theta$ with gradient descent with specified learning rate

and, after all iterations, returns the final estimate of $\theta$.
"""

# ╔═╡ 3c657688-617c-11eb-2655-415562d132bb
function train_nn_w_var_reg(target_f, θ_init; bs= 100, lr = 1e-4, iters=10000)
    θ_curr = θ_init
    for i in 1:iters
      x,y = #TODO
      @info "loss: $()" #TODO: log loss
      grad_θ = #TODO compute gradients
      θ_curr = #TODO gradient descent
    end
    return θ_curr
end

# ╔═╡ 44cdb444-617c-11eb-1c69-8bb0197c9c32
md"""
### Learn model with input-dependent variance

 For each target function, start with an initialization of the network parameters, $\theta$,
    learn an estimate for $\theta_\text{learned}$ by gradient descent.
    Then plot a $n=1000$ sample of the dataset and the learned regression model with shaded uncertainty bounds corresponding to plus/minus one standard deviation given by the variance of the predictive distribution at each input location 
    (output by the neural network).
    (Hint: `ribbon` argument for shaded uncertainty bounds can accept a vector of $\sigma$)

Note: Learning the variance is tricky, and this may be unstable during training. There are some things you can try:
* Adjusting the hyperparameters like learning rate and batch size
* Train for more iterations
* Try a different random initialization, like sample random weights and bias matrices with lower variance.
    
For this question **you will not be assessed on the final quality of your model**.
Specifically, if you fails to train an optimal model for the data that is okay. 
You are expected to learn something that is somewhat reasonable, and **demonstrates that this model is training and learning variance**.

If your implementation is correct, it is possible to learn a reasonable model with fewer than 10 minutes of training on a laptop CPU. 
The default hyperparameters should help, but may need some tuning.

"""

# ╔═╡ 74bd837a-617c-11eb-3716-07cd84f5f4ac
#TODO: For each target function
θ_init  #TODO

# ╔═╡ 788a21fe-617c-11eb-11fa-4dc3da665951
θ_learned #TODO

# ╔═╡ 79674636-617c-11eb-0213-fb99e78e9f1d
md"""
### Plot model
"""

# ╔═╡ 84c933cc-617c-11eb-3f49-65f335a05486
#TODO: plot data samples and learned regression

# ╔═╡ 8c3f1d38-617c-11eb-2820-f32c96e276c6
md"""
### Spend time making it better (optional)

If you would like to take the time to train a very good model of the data (specifically for target functions 2 and 3) with a neural network
that outputs both mean and $\log \sigma$ you can do this, but it is not necessary to achieve full marks.

You can try
* Using a more stable optimizer, like Adam. You may import this from a library.
* Increasing the expressivity of the neural network, increase the number of layers or the dimensionality of the hidden layer.
* Careful tuning of hyperparameters, like learning rate and batchsize.
"""

# ╔═╡ 8c5b76a4-6177-11eb-223a-6b656e173ffb
@testset "sample dimensions are correct" begin
  m = 1 # dimensionality
  n = 200 # batch-size
  for target_f in (target_f1, target_f2, target_f3)
    x,y = sample_batch(target_f,n)
    @test size(x) == (m,n)
    @test size(y) == (n,)
  end
end

# ╔═╡ bfa935e8-617a-11eb-364c-93b89a9b3e23
@testset "neural net mean vector output" begin
n = 100
x,y = sample_batch(target_f1,n)
μ = neural_net(x,θ)
@test size(μ) == (n,)
end

# ╔═╡ e8da3e14-6178-11eb-2058-6d5f4a77378b
@testset "Gaussian log likelihood" begin
using Distributions: pdf, Normal
# Scalar mean and variance
x = randn()
μ = randn()
σ = rand()
@test size(gaussian_log_likelihood(μ,σ,x)) == () # Scalar log-likelihood
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x)) # Correct Value
# Vector valued x under constant mean and variance
x = randn(100)
μ = randn()
σ = rand()
@test size(gaussian_log_likelihood.(μ,σ,x)) == (100,) # Vector of log-likelihoods
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal(μ,σ),x)) # Correct Values
# Vector valued x under vector valued mean and variance
x = randn(10)
μ = randn(10)
σ = rand(10)
@test size(gaussian_log_likelihood.(μ,σ,x)) == (10,) # Vector of log-likelihoods
@test gaussian_log_likelihood.(μ,σ,x) ≈ log.(pdf.(Normal.(μ,σ),x)) # Correct Values
end

# ╔═╡ d4a5a4e6-6179-11eb-0596-bb8ddc6027fb
@testset "Gradients wrt parameter" begin
β_test = randn()
σ_test = rand()
x,y = sample_batch(target_f1,100)
ad_grad = #TODO
hand_derivative = #TODO
@test ad_grad[1] ≈ hand_derivative
end

# ╔═╡ d9c41410-617b-11eb-2542-6da3f6bd0498

@testset "neural net mean and logsigma vector output" begin
n = 100
x,y = sample_batch(target_f1,n)
μ, logσ = neural_net_w_var(x,θ)
@test size(μ) == (n,)
@test size(logσ) == (n,)
end

# ╔═╡ 241f7756-6178-11eb-19a6-c516934c4208
n=1000 # batch_size

# ╔═╡ Cell order:
# ╟─3b614428-6176-11eb-27af-1d611c78a404
# ╟─4c8b7872-6176-11eb-3c4c-117dfe425f8a
# ╟─16c9fa00-6177-11eb-27b4-175a938132af
# ╠═2038c102-6177-11eb-047e-95829eaf59b7
# ╠═361f0788-6177-11eb-35a9-d12176832720
# ╠═361f450e-6177-11eb-1320-9b16d3cbe14c
# ╠═361f9f40-6177-11eb-0792-3f7214a6dafe
# ╠═61d33d5e-6177-11eb-17be-ff5d831900fa
# ╠═648df4f8-6177-11eb-225e-8f9955713b67
# ╠═6ac7bb10-6177-11eb-02bc-3513af63a9b9
# ╠═8625e7ba-6177-11eb-0f73-f14acfe6a2ea
# ╠═8c5b76a4-6177-11eb-223a-6b656e173ffb
# ╟─91b725a8-6177-11eb-133f-cb075b2c50dc
# ╠═c1c7bd02-6177-11eb-3073-d92dde78c4bb
# ╠═2be4e23c-6178-11eb-3019-27d3ac7e6d2e
# ╠═2be53474-6178-11eb-2b75-1f1c3187770f
# ╠═2be5b9b4-6178-11eb-3e01-61fea97cdfa6
# ╠═2bf298d2-6178-11eb-23e3-77a2347ed0a2
# ╠═2bf9191e-6178-11eb-2db3-f358d690b2a3
# ╠═2bffc028-6178-11eb-0718-199b410131eb
# ╠═f6999bb8-6177-11eb-3a0e-35e7a718e0ab
# ╠═f9f79404-6177-11eb-2d31-857cb99cf7db
# ╠═1cfe23f0-6178-11eb-21bf-8d8a46b0d7c6
# ╠═241f7756-6178-11eb-19a6-c516934c4208
# ╠═241fda48-6178-11eb-1775-232a4b55d258
# ╠═242049d0-6178-11eb-2572-ad3f6ea43127
# ╠═242af220-6178-11eb-3bfe-6bcf3e814ff1
# ╠═24329fca-6178-11eb-0bdc-45e28e499635
# ╠═24330276-6178-11eb-0a8f-0f49cb7d57fa
# ╠═243e38ee-6178-11eb-1c09-71018451b8f7
# ╠═4f04aca2-6178-11eb-0e59-df88659f11c1
# ╠═6222e9e0-6178-11eb-3448-57e951600ef9
# ╠═705a7d28-6178-11eb-280d-315614ad9080
# ╠═742afde4-6178-11eb-08b6-e77847b0aa40
# ╠═7f61ff44-6178-11eb-040b-2f19e52ebaf5
# ╠═baf4cbea-6178-11eb-247f-4961c91da925
# ╠═ccedbd3e-6178-11eb-2d26-01016c9dea4b
# ╠═e8da3e14-6178-11eb-2058-6d5f4a77378b
# ╠═045fd22a-6179-11eb-3901-8b327ed3b7a0
# ╠═2b7be59c-6179-11eb-3fde-01eac8b60a9d
# ╠═314fd0fa-6179-11eb-0d0c-a7efdfa33c65
# ╠═5b1dec3c-6179-11eb-2968-d361e1c18cc8
# ╠═6cdd6d12-6179-11eb-1115-cd43e97e9a60
# ╠═97606f8a-6179-11eb-3dbb-af72ec35e4a1
# ╠═abe7efdc-6179-11eb-14ed-a3b0462cc2f0
# ╠═d2cb6eda-6179-11eb-3e11-41b238bd071d
# ╠═d4a5a4e6-6179-11eb-0596-bb8ddc6027fb
# ╠═d9d4d6da-6179-11eb-0165-95d79e1ab92d
# ╠═033afc0c-617a-11eb-32a9-f3f467476a0a
# ╠═217b1466-617a-11eb-19c6-2f29ef5d3576
# ╠═449d93e6-617a-11eb-3289-a931618f4bba
# ╠═4adae2ac-617a-11eb-0ed7-cdb1985fad44
# ╠═4d628d18-617a-11eb-1944-c922f564a1f3
# ╠═4adb7980-617a-11eb-041d-214170284e3a
# ╠═78cc51f0-617a-11eb-3408-83b8d44df832
# ╟─8d762018-617a-11eb-3def-27d3959bb155
# ╠═a95e69f2-617a-11eb-3034-e12b72492357
# ╠═bfa8826a-617a-11eb-2be8-a3fe151a32cb
# ╠═ca10a7fa-617a-11eb-1568-4724c3686b01
# ╠═bfa935e8-617a-11eb-364c-93b89a9b3e23
# ╠═e3a14bac-617a-11eb-2155-ff812544df13
# ╠═f65af5b8-617a-11eb-3865-7fa972a6a821
# ╠═0f355c54-617b-11eb-2768-7b8066538440
# ╠═3ca4bbbc-617b-11eb-36e4-ad24b86bfd19
# ╠═429d9e76-617b-11eb-161c-2b16653d2b0c
# ╠═51a22d88-617b-11eb-27cd-158ba4b38acc
# ╠═5b9a5c98-617b-11eb-2ede-9dbbeb8ea32d
# ╠═89cdc082-617b-11eb-3491-453302b03caa
# ╠═b04e11d0-617b-11eb-35a4-b5aeeafb8570
# ╠═c7ae64a6-617b-11eb-10b8-712abc9dc2a6
# ╠═d6ad1a06-617b-11eb-224c-0307a6e6f80a
# ╠═c82cff46-617b-11eb-108a-a9576c728328
# ╠═d9c41410-617b-11eb-2542-6da3f6bd0498
# ╠═e7cddc4e-617b-11eb-3b2c-f7c06fc61fd5
# ╠═0b7d0ac0-617c-11eb-24c5-b3126ee28f5a
# ╠═128daf4a-617c-11eb-3c62-1b61708169e0
# ╠═3c657688-617c-11eb-2655-415562d132bb
# ╠═44cdb444-617c-11eb-1c69-8bb0197c9c32
# ╠═74bd837a-617c-11eb-3716-07cd84f5f4ac
# ╠═788a21fe-617c-11eb-11fa-4dc3da665951
# ╠═79674636-617c-11eb-0213-fb99e78e9f1d
# ╠═84c933cc-617c-11eb-3f49-65f335a05486
# ╠═8c3f1d38-617c-11eb-2820-f32c96e276c6
