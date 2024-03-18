import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.gp.parameterized import Parameterized
from pyro.contrib.gp.util import param_with_module_name
from neural_ode import NeuralODE

# Generate synthetic data
def generate_data(num_data_points):
    true_slope = 1.5
    true_intercept = 2.0
    x = torch.linspace(0, 10, num_data_points)
    y = true_slope * x + true_intercept + torch.randn(num_data_points) * 2.0
    return x, y

# Define the Bayesian linear regression model with Neural ODE
class NeuralODEModel(pyro.nn.PyroModule):
    def __init__(self):
        super().__init__()
        self.linear = pyro.nn.PyroLinear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Define the CJS model
class CJSModel(Parameterized):
    def __init__(self):
        super().__init__()
        self.lengthscale = pyro.nn.PyroParam(torch.tensor(1.0), constraint=dist.constraints.positive)
        self.intensity = pyro.nn.PyroParam(torch.tensor(1.0), constraint=dist.constraints.positive)

    def model(self, x, y):
        lengthscale = pyro.sample(param_with_module_name(self, "lengthscale"), dist.LogNormal(0, 1))
        intensity = pyro.sample(param_with_module_name(self, "intensity"), dist.LogNormal(0, 1))
        kernel = pyro.contrib.gp.kernels.RBF(input_dim=1, lengthscale=lengthscale)
        cjs = pyro.contrib.gp.models.CJSModel(kernel, num_points=len(x), intensity=intensity)
        cjs.wrap_model(x, y)

# Define the guide (variational distribution) for inference
def guide(x, y):
    pyro.module("model", NeuralODEModel())
    solver = NeuralODE(NeuralODEModel())
    h0 = torch.zeros(1, 1)  # Initial hidden state for ODE solver
    t = torch.linspace(0, 1, steps=len(x))  # Time steps for ODE solver
    ode_output = solver.odeint(x, h0, t)
    predicted_y = ode_output.squeeze(-1)

    # CJS model guide
    cjs_model = CJSModel()
    cjs_model.guide(x, y)

# Perform inference using SVI (Stochastic Variational Inference)
def inference(x, y, num_steps):
    pyro.clear_param_store()
    optimizer = Adam({"lr": 0.05})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step(x, y)
        if step % 100 == 0:
            print(f'Step {step}: loss = {loss:.4f}')

if __name__ == "__main__":
    # Generate synthetic data
    x_data, y_data = generate_data(100)

    # Perform inference
    inference(x_data, y_data, 1000)

    # Get the learned parameters
    lengthscale = pyro.param(param_with_module_name("model$$$cjs_model", "lengthscale")).item()
    intensity = pyro.param(param_with_module_name("model$$$cjs_model", "intensity")).item()
    print(f'Learned lengthscale: {lengthscale:.4f}')
    print(f'Learned intensity: {intensity:.4f}')
