
import numpy as np

"""

Vectorized implementation of Glicko2 inference algorithm,
for a single period.

"""

#Step 1: Define functions

def g(phi):

    return 1/np.sqrt(1 + 3*phi**2/np.pi**2)

def e(mu, mu_j, phi_j):

    n_player = mu.shape[0]

    mu_ = mu.repeat(n_player).reshape(n_player,n_player)

    mu_j = mu_j.reshape(1,-1)

    return 1/(1 + np.exp(-g(phi_j**2)*(mu_-mu_j)))

def phi_(phi_prime, nu):

    return (1/phi_prime**2 + 1/nu**2)**-1

def f(x, delta, phi, nu, a, tau):

    res = np.exp(x)*(delta**2 - phi**2 - nu - np.exp(x))

    res /= 2 * (phi**2 + nu + np.exp(x))

    res -= (x - a) / tau**2

    return res

#Step 2: Initialize

eps = 1e-6

mu = np.random.randn(
    data['n_player'],
    )

phi = abs(np.random.randn(
    data['n_player'],
    ))

sigma = 0.06 * np.ones(data['n_player'])

#Simulated data:
scores = np.random.randint(low=0,high=1,size=(
    data['n_player'],
    data['n_player']
    )
    )

tau = 0.2

#Step 3: Compute \nu (v)

nu = np.sum(g(phi)**2*e(mu, mu, phi)*(1- e(mu, mu, phi)),1)**-1

#Step 4: Compute \delta (Δ)

delta = nu*np.sum(g(phi)*(scores - e(mu, mu, phi)), 1)

#Step 5: Compute \sigma' (σ')

A = 2*np.log(sigma)

B = np.zeros(data['n_player'])

#Step 5.2:

for i in range(len(delta)):
    
    if delta[i] > phi[i]**2 + nu[i]:
        
        B[i] = np.log(delta[i]**2 - phi[i]**2 - nu[i])
        
    elif delta[i] <= phi[i]**2 + nu[i] > 1:

         k = 1

         while f(A[i] - k*tau, delta[i], phi[i], nu[i], A[i], tau) < 0:

             k += 1

         B = A - k*tau

#Step 5.3, 5.4:

while sum(A - B > eps) > 1:

    fa = f(A, delta, phi, nu, A, tau)

    fb = f(B, delta, phi, nu, A, tau)

    C = A + (A - B) * fa / (fb - fa)

    fc = f(C, delta, phi, nu, A, tau)

    if sum(fc*fb < 0) > 1:

        A = B

        fa = fb

    else:

        fa /= 2

    B = C

    fb = fc

#Step 5.5

sigma_prime = np.exp(A/2)

#Step 6:

phi_star = np.sqrt(sigma_prime**2 + phi**2)

#Step 7:

phi_prime = 1/np.sqrt(1/phi_star**2 + 1/nu)

mu_prime = mu + phi_prime**2 * np.sum(g(phi)**2*e(mu, mu, phi)*(1- e(mu, mu, phi)),1)**-1
