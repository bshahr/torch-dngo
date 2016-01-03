--[[
Quick and dirty torch implementation of the network featured in
Scalable Bayesian optimization using deep neural networks by
Snoek et al. (ICML 2015).

This is simply a lua transcription of the accompanying itorch
notebook and should not require any non-standard lua libraries.

Bobak Shahriari, 2015
--]]

require 'nn'
require 'optim'

ninputs = 1
nhidden = 50
noutputs = 1

-- define the DNGO network
model = nn.Sequential()
model:add(nn.Linear(ninputs, nhidden))
model:add(nn.Tanh())    -- can also try: model:add(nn.ReLU())
model:add(nn.Linear(nhidden, nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden, nhidden))
model:add(nn.Tanh())
model:add(nn.Linear(nhidden, noutputs))

-- define MSE loss function
loss = nn.MSECriterion()

noise = 0.2
ntrain = 50
ntest = 1000
nperbatch = 20

function objective(x)
    return torch.sin(torch.mul(x, 7)) + torch.cos(torch.mul(x, 17))
end

-- generate training data and fix test inputs
-- the trailing underscore indicates that these are flattened arrays
xtrain_ = torch.rand(ntrain)
ytrain_ = objective(xtrain_) + torch.randn(ntrain):mul(noise)
xtest_ = torch.linspace(-1., 2., ntest)

-- maintain square shaped data
-- because input and output are univariate so indexing returns
-- a number when often a Tensor is expected.
xtrain = xtrain_:reshape(ntrain, 1)
ytrain = ytrain_:reshape(ntrain, 1)
xtest = xtest_:reshape(ntest, 1)

function plot_objective(x, plot)
    -- plot objective at points x
    local y = objective(x)
    local plot = plot or itorch.Plot()
    plot:line(x, y, 'black', 'true'):redraw()
    return plot
end

function plot_data(x, y, plot)
    -- plot data points (x, y)
    local plot = plot or itorch.Plot():draw()
    plot:circle(x, y, 'red', 'data'):redraw()
    return plot
end

-- store pointer to weights and their gradients
weights, dweights = model:getParameters()


function eval(w)
    -- evaluate error and its gradient with respect to weights
    if w ~= weights then
        weights:copy(w)
    end
    
    dweights:zero()

    -- compute prediction and error
    local outputs = model:forward(xtrain)
    local err = loss:forward(outputs, ytrain)
        
    -- backpropagate gradient
    local dloss = loss:backward(outputs, ytrain)
    model:backward(xtrain, dloss)  -- alters dweights
    
    return err, dweights
end

-- stochastic gradient descent hyperparameters
config = {
    learningRate = 0.1,
    learningRateDecay = 5e-7,
    momentum = 0.1,
    -- weightDecay = 0.01,
}

function train(nepochs)
    -- trains model for nepochs
    local epoch = 1
    while epoch < nepochs do
        epoch = epoch + 1
        optim.sgd(eval, weights, config)
    end
end

nplots = 6
colors = {'blue', 'green', 'red', 'purple', 'orange', 'magenta', 'cyan'}
nbase = 400

plot = itorch.Plot()
plot_objective(xtest_, plot)
plot_data(xtrain_, ytrain_, plot)
train(nbase)
ypred_ = model:forward(xtest):reshape(ntest)
plot:line(xtest_, ypred_, colors[1], tostring(nbase))

plot:title('Neural network fit after N epochs')

-- train and plot
for i = 1, nplots-1 do
    local nepochs = nbase * 2^(i-1)
    local label = tostring(nbase * 2^i)
    
    train(nepochs)
    ypred_ = model:forward(xtest):reshape(ntest)
    plot:line(xtest_, ypred_, colors[i+1], label)
    plot:legend(true):redraw()
end

function getFeatures(inputs)
    local ninputs = inputs:size()[1]
    local nlayers = #model.modules
    local nbasis = model.modules[#model.modules].gradInput:size()[1]
    local phi = torch.Tensor(ninputs, nbasis)
    
    for i = 1, ninputs do
        model:forward(inputs[i])
        phi[i] = model.modules[nlayers-2].output
    end
    
    return phi
end

function plotFeatures(inputs, plot)
    local plot = plot or itorch.Plot()
    phi = getFeatures(inputs:reshape(ntest, 1))
    
    for i = 1, phi:size()[2] do
        plot:line(inputs, phi:t()[i]):redraw()
    end
    
    return plot
end

-- uncomment to plot the learned basis functions
-- plot = itorch.Plot():title('Learned basis functions')
-- plotFeatures(xtest, plot)

function bayesianLinearRegression(alpha, beta)
    local alpha = alpha or 0.1                  -- corresponds to noise sigma
    local beta = beta or 0.1                    -- corresponds to kernel amplitude squared
    
    local nbasis = model.modules[#model.modules].gradInput:size()[1]

    local phi = getFeatures(xtest)
    local phi_train = getFeatures(xtrain)

    -- computes `alpha^2 * I + beta * phi.T phi`
    K = torch.addmm(alpha^2, torch.eye(nbasis), beta, phi_train:t(), phi_train)

    -- invert matrix
    Kphi = torch.mm(torch.inverse(K), phi:t())
    -- could also run the following if `nbasis` is large
    -- Kphi = torch.gels(phi:t(), K)

    -- compute predictive mean and variance
    mu = torch.mv(Kphi:t(), torch.mv(phi_train:t(), ytrain_)):mul(beta)
    s2 = torch.cmul(phi:t(), Kphi):sum(1):add(1/beta)

    return mu, s2
end

mu, s = bayesianLinearRegression(0.2, 4)            -- hypers set according to true objective
s:sqrt():mul(2)

plot = itorch.Plot()
plot:title('Bayesian linear regression with learned features')
plot_objective(xtest_, plot)
plot_data(xtrain_, ytrain_, plot)

plot:line(xtest_, mu, 'red', 'posterior')
plot:line(xtest_, mu - s, 'blue', 'credible')
plot:line(xtest_, mu + s, 'blue', 'interval')
plot:legend(true):redraw()
