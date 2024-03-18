import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Generate synthetic data
def generate_data(num_data_points):
    true_slope = 1.5
    true_intercept = 2.0
    x = torch.linspace(0, 10, num_data_points)
    y = true_slope * x + true_intercept + torch.randn(num_data_points) * 2.0
    return x, y

# Define the Bayesian linear regression model
def model(x, y):
    slope = pyro.sample('slope', dist.Normal(0, 10))
    intercept = pyro.sample('intercept', dist.Normal(0, 10))
    y_pred = slope * x + intercept
    with pyro.plate('data', len(x)):
        pyro.sample('obs', dist.Normal(y_pred, 2.0), obs=y)

# Define the guide (variational distribution) for inference
def guide(x, y):
    slope_loc = pyro.param('slope_loc', torch.tensor(0.0))
    slope_scale = pyro.param('slope_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
    intercept_loc = pyro.param('intercept_loc', torch.tensor(0.0))
    intercept_scale = pyro.param('intercept_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
    slope = pyro.sample('slope', dist.Normal(slope_loc, slope_scale))
    intercept = pyro.sample('intercept', dist.Normal(intercept_loc, intercept_scale))

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
    slope = pyro.param('slope_loc').item()
    intercept = pyro.param('intercept_loc').item()
    print(f'Learned slope: {slope:.4f}')
    print(f'Learned intercept: {intercept:.4f}')
