import numpy as np
import math

# Question 1 — Exponential Distribution

def exponential_pdf(x, lam):
    if x < 0:
        return 0
    return lam * math.exp(-lam * x)
def probability_between(a, b, lam, steps=100000):
    x_vals = np.linspace(a, b, steps)
    pdf_vals = lam * np.exp(-lam * x_vals)
    dx = (b - a) / steps
    probability = np.sum(pdf_vals) * dx
    return probability
def simulate_exponential_probability(a, b, lam, samples=100000):
    data = np.random.exponential(scale=1/lam, size=samples)
    count = np.sum((data > a) & (data < b))
    return count / samples

# Question 2 — Bayesian Classification (Gaussian Model)

def gaussian_pdf(x, mean, variance):
    coefficient = 1 / math.sqrt(2 * math.pi * variance)
    exponent = math.exp(-((x - mean) ** 2) / (2 * variance))
    return coefficient * exponent

def compute_likelihoods(x, meanA, varA, meanB, varB):
    likelihood_A = gaussian_pdf(x, meanA, varA)
    likelihood_B = gaussian_pdf(x, meanB, varB)
    return likelihood_A, likelihood_B
def bayes_posterior_B(x, meanA, varA, meanB, varB, PA, PB):
    likelihood_A, likelihood_B = compute_likelihoods(x, meanA, varA, meanB, varB)
    numerator = likelihood_B * PB
    denominator = (likelihood_A * PA) + (likelihood_B * PB)

    return numerator / denominator
def simulate_swimmers_posterior(x_obs, meanA, varA, meanB, varB, PA, PB, samples=100000):
    labels = np.random.choice(['A', 'B'], size=samples, p=[PA, PB])
    times = []

    for label in labels:
        if label == 'A':
            times.append(np.random.normal(meanA, math.sqrt(varA)))
        else:
            times.append(np.random.normal(meanB, math.sqrt(varB)))
    times = np.array(times)
    tolerance = 0.5
    mask = np.abs(times - x_obs) < tolerance
    selected_labels = labels[mask]

    if len(selected_labels) == 0:
        return 0
    return np.sum(selected_labels == 'B') / len(selected_labels)
