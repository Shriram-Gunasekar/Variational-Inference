import torch
import pyro
import pyro.distributions as dist

# Define the probabilistic model
def coin_flip():
    # Define the prior probability for the bias of the coin
    bias = pyro.sample('bias', dist.Beta(10, 10))
    # Use the biased coin to generate a sample
    result = pyro.sample('result', dist.Bernoulli(bias))
    return result.item()

# Perform inference
def inference():
    # Initialize the counter for heads
    heads = 0
    # Run the model multiple times to collect samples
    for _ in range(1000):
        outcome = coin_flip()
        heads += outcome
    # Calculate the probability of heads
    probability_heads = heads / 1000.0
    print(f"Probability of heads: {probability_heads}")

if __name__ == "__main__":
    inference()
