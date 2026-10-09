
import gpytorch
import torch
import tqdm
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

class twoBranchMFModel(torch.nn.Module):

    def __init__(self, data_dim:int=1, hidden_dim:int=128):
        super(twoBranchMFModel, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(data_dim-1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )


        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(data_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.LeakyReLU()
        )


        self.linear3 = torch.nn.Linear((hidden_dim), 1)

    def forward(self, x):


        z2 = self.branch2(x)


        y = z2
        return y

class MFGPRegressionModel(SingleTaskGP):
        def __init__(self, train_x, train_y, likelihood, trained_branch_mlp:torch.nn.Module=None):
            super(MFGPRegressionModel, self).__init__(train_x, train_y, likelihood)
            twoBranchMFModel = twoBranchMFModel(train_x.size(-1)) if trained_branch_mlp is None else trained_branch_mlp
            class MFFeastureExtractor(torch.nn.Module):
                def __init__(self, model):
                    super(MFFeastureExtractor, self).__init__()
                    self.branch1 = model.branch1
                    self.branch2 = model.branch2

                def forward(self, x):
                    z1 = self.branch1(x[:, :-1])
                    z2 = self.branch2(x)
                    z = z1 + z2
                    return z

            self.feature_extractor = MFFeastureExtractor(model=twoBranchMFModel)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )


            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            self.to(train_x)

        def forward(self, x):

            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)
            return super().forward(projected_x)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim=1):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 128, dtype=torch.float64))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(128, max(data_dim-1, 2), dtype=torch.float64))

class RBFGPRegressionModel(SingleTaskGP):
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        super(RBFGPRegressionModel, self).__init__(train_x, train_y, likelihood, outcome_transform=Standardize(m=1))
        self.mean_module = mean
        self.covar_module = covar

class GPRegressionModel(SingleTaskGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood, outcome_transform=Standardize(m=1))
            feature_extractor = LargeFeatureExtractor(train_x.size(-1))
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=max(train_x.size(-1)-1,2))
            )
            self.feature_extractor = feature_extractor


            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            self.to(train_x)

        def forward(self, x):

            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)
            return super().forward(projected_x)

def spectral_norm(weight, n_power_iterations=1):
    w = weight
    height = w.size(0)
    u = torch.randn(height).to(w)

    for _ in range(n_power_iterations):
        v = torch.mv(torch.t(w), u)
        v = v / torch.norm(v)
        u = torch.mv(w, v)
        u = u / torch.norm(u)

    sigma = torch.dot(u, torch.mv(w, v))
    return sigma

class SpectralRegularizer:
    def __init__(self, model, coeff=1e-4, n_power_iterations=10):
        self.model = model
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations

    def __call__(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                sigma = spectral_norm(param, self.n_power_iterations)
                reg_loss += sigma

        return self.coeff * ((reg_loss - 1)**2)

def init_model_srdk(
    train_x,
    train_obj,
    coeff=1e-4,
    training_iter=1000,
    lr=1e-2,
    power_iterations=100,
    verbose=True,
    **kwargs,
):
    likelihood_c = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_obj, likelihood_c)
    model.to(train_x)


    model.train()
    likelihood_c.train()

    optimizer_c = torch.optim.Adam(
        [
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ],
        lr=lr,
    )


    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    num_epochs = training_iter
    train_iterator_c = (
        tqdm.tqdm(range(num_epochs), desc="Training Deep Kernel GP Combined")
        if verbose else range(num_epochs)
    )

    for epoch in train_iterator_c:
        optimizer_c.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_obj.squeeze(-1))

        reg_loss = SpectralRegularizer(
            model,
            coeff=coeff,
            n_power_iterations=power_iterations,
        )()

        total_loss = loss + reg_loss
        total_loss.backward()

        if verbose:
            train_iterator_c.set_postfix(
                nll=loss.item(),
                reg=reg_loss.item(),
                total=total_loss.item(),
            )

        optimizer_c.step()

    return mll, model

def convert_to_nn_and_base_kernel(model, x_candidates):
    feasture_extractor = model.feature_extractor
    feasture_extractor.eval()
    projected_x = feasture_extractor(x_candidates)
    projected_x = model.scale_to_bounds(projected_x)
    base_model = RBFGPRegressionModel(feasture_extractor(model.train_inputs[0]), model.train_targets.reshape(-1,1), model.likelihood, model.mean_module, model.covar_module)
    return base_model, projected_x
