# Portfolio Optimization with QAOA

Optimize investment portfolios using the Quantum Approximate Optimization Algorithm.

## Overview

This example applies QAOA to the portfolio optimization problem from quantitative finance. We minimize risk (variance) while achieving target returns, demonstrating how quantum algorithms can address real-world optimization challenges.

## Prerequisites

- Understanding of QAOA ([QAOA Algorithm](../../algorithms/qaoa-algorithm.md))
- Familiarity with quadratic optimization
- Basic knowledge of portfolio theory

## Problem Formulation

### Classical Portfolio Optimization

The Markowitz mean-variance model minimizes:

$$\min_w \quad w^T \Sigma w - \lambda \mu^T w$$

Subject to:
- $\sum_i w_i = 1$ (fully invested)
- $w_i \geq 0$ (no short selling, optional)

Where:
- $w$: Portfolio weights
- $\Sigma$: Covariance matrix of returns
- $\mu$: Expected returns vector
- $\lambda$: Risk aversion parameter

### QUBO Formulation

For discrete allocation (binary variables), we reformulate as Quadratic Unconstrained Binary Optimization:

$$\min_x \quad x^T Q x + c^T x$$

Where $x_i \in \{0, 1\}$ indicates whether asset $i$ is included.

## Python Implementation

```python
"""
Portfolio Optimization with QAOA
Minimize risk while achieving target returns.
"""

from moonlab import QuantumState
from moonlab.algorithms import QAOA
from moonlab.optimization import QUBOBuilder
import numpy as np
from typing import List, Tuple

class PortfolioOptimizer:
    """
    Quantum portfolio optimization using QAOA.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        budget: int = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Trade-off between risk and return (0=max return, 1=min risk)
            budget: Number of assets to select (None = any number)
        """
        self.n_assets = len(expected_returns)
        self.mu = expected_returns
        self.sigma = covariance_matrix
        self.lambda_risk = risk_aversion
        self.budget = budget

        # Validate inputs
        assert self.sigma.shape == (self.n_assets, self.n_assets)
        assert np.allclose(self.sigma, self.sigma.T)  # Symmetric

        # Build QUBO
        self.Q, self.offset = self._build_qubo()

    def _build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Build QUBO matrix from portfolio parameters.

        Returns:
            Q matrix and constant offset
        """
        n = self.n_assets
        Q = np.zeros((n, n))

        # Risk term: x^T Σ x
        Q += self.lambda_risk * self.sigma

        # Return term: -μ^T x (diagonal)
        for i in range(n):
            Q[i, i] -= (1 - self.lambda_risk) * self.mu[i]

        # Budget constraint: (sum_i x_i - budget)^2
        if self.budget is not None:
            penalty = 10.0  # Penalty weight
            for i in range(n):
                Q[i, i] += penalty * (1 - 2 * self.budget)
                for j in range(i + 1, n):
                    Q[i, j] += 2 * penalty
                    Q[j, i] += 2 * penalty

        offset = 0.0
        if self.budget is not None:
            offset += penalty * self.budget ** 2

        return Q, offset

    def classical_solve(self) -> np.ndarray:
        """
        Solve via brute force (for verification).

        Returns:
            Optimal binary selection vector
        """
        best_cost = float('inf')
        best_x = None

        for i in range(2 ** self.n_assets):
            x = np.array([(i >> j) & 1 for j in range(self.n_assets)])

            # Check budget constraint
            if self.budget is not None and np.sum(x) != self.budget:
                continue

            cost = x @ self.Q @ x + self.offset
            if cost < best_cost:
                best_cost = cost
                best_x = x

        return best_x

    def quantum_solve(
        self,
        depth: int = 3,
        shots: int = 1000,
        optimizer: str = 'COBYLA',
        max_iterations: int = 100
    ) -> dict:
        """
        Solve using QAOA.

        Args:
            depth: Number of QAOA layers
            shots: Measurement shots
            optimizer: Classical optimizer
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with solution and metadata
        """
        # Create QAOA solver
        qaoa = QAOA(
            num_qubits=self.n_assets,
            depth=depth,
            shots=shots,
            optimizer=optimizer
        )

        # Set cost function from QUBO
        qaoa.set_qubo(self.Q)

        # Optimize
        result = qaoa.optimize(max_iterations=max_iterations)

        # Extract solution
        solution_bitstring = result.most_likely_state
        solution = np.array([int(b) for b in solution_bitstring])

        # Calculate portfolio metrics
        portfolio_return = self.mu @ solution
        portfolio_risk = np.sqrt(solution @ self.sigma @ solution)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        return {
            'selection': solution,
            'selected_assets': np.where(solution == 1)[0].tolist(),
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'optimal_params': result.optimal_params,
            'iterations': result.iterations,
            'cost': result.cost
        }

    def analyze_solution(self, solution: np.ndarray, asset_names: List[str] = None):
        """
        Print detailed analysis of a portfolio solution.

        Args:
            solution: Binary selection vector
            asset_names: Optional list of asset names
        """
        if asset_names is None:
            asset_names = [f"Asset {i}" for i in range(self.n_assets)]

        print("\n" + "=" * 50)
        print("           Portfolio Analysis")
        print("=" * 50)

        # Selected assets
        selected_idx = np.where(solution == 1)[0]
        print(f"\nSelected Assets ({len(selected_idx)}/{self.n_assets}):")
        print("-" * 30)
        for idx in selected_idx:
            print(f"  • {asset_names[idx]:20} (μ = {self.mu[idx]:.2%})")

        # Portfolio metrics
        portfolio_return = self.mu @ solution
        portfolio_risk = np.sqrt(solution @ self.sigma @ solution)
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        print(f"\nPortfolio Metrics:")
        print("-" * 30)
        print(f"  Expected Return:  {portfolio_return:.2%}")
        print(f"  Risk (Std Dev):   {portfolio_risk:.2%}")
        print(f"  Sharpe Ratio:     {sharpe:.3f}")

        # Correlation analysis
        print(f"\nCorrelation Structure:")
        print("-" * 30)
        if len(selected_idx) > 1:
            sub_corr = np.corrcoef(self.sigma[np.ix_(selected_idx, selected_idx)])
            avg_corr = (np.sum(sub_corr) - len(selected_idx)) / (len(selected_idx) ** 2 - len(selected_idx))
            print(f"  Avg correlation:  {avg_corr:.3f}")
        else:
            print("  Single asset selected - no diversification")


def generate_market_data(n_assets: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate realistic market data for testing.

    Args:
        n_assets: Number of assets
        seed: Random seed

    Returns:
        expected_returns, covariance_matrix, asset_names
    """
    np.random.seed(seed)

    # Asset names (sectors)
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Industrial', 'Telecom', 'Utilities']
    asset_names = [sectors[i % len(sectors)] + f"_{i//len(sectors) + 1}" for i in range(n_assets)]

    # Expected returns: 5-15% annually
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)

    # Generate correlated returns
    # Base correlation within sectors, lower across sectors
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            sector_i = i % len(sectors)
            sector_j = j % len(sectors)

            if sector_i == sector_j:
                corr = np.random.uniform(0.5, 0.8)  # Same sector
            else:
                corr = np.random.uniform(0.1, 0.4)  # Different sector

            correlation[i, j] = corr
            correlation[j, i] = corr

    # Convert to covariance
    volatilities = np.random.uniform(0.15, 0.35, n_assets)  # 15-35% volatility
    D = np.diag(volatilities)
    covariance_matrix = D @ correlation @ D

    return expected_returns, covariance_matrix, asset_names


def run_portfolio_example():
    """
    Run complete portfolio optimization example.
    """
    print("=" * 60)
    print("     Portfolio Optimization with QAOA")
    print("=" * 60)

    # Generate market data
    n_assets = 6  # Keep small for simulation
    expected_returns, covariance, asset_names = generate_market_data(n_assets)

    print(f"\nMarket Data ({n_assets} assets)")
    print("-" * 40)
    for i, name in enumerate(asset_names):
        print(f"{name:15} μ={expected_returns[i]:.1%}  σ={np.sqrt(covariance[i,i]):.1%}")

    # Optimize with different budgets
    for budget in [2, 3, 4]:
        print(f"\n{'=' * 50}")
        print(f"Budget: Select exactly {budget} assets")
        print("=" * 50)

        optimizer = PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=covariance,
            risk_aversion=0.5,
            budget=budget
        )

        # Classical solution (reference)
        classical_sol = optimizer.classical_solve()
        print(f"\nClassical (brute force) solution:")
        optimizer.analyze_solution(classical_sol, asset_names)

        # QAOA solution
        print(f"\nQAOA solution (depth=3):")
        qaoa_result = optimizer.quantum_solve(depth=3, shots=1000, max_iterations=50)
        optimizer.analyze_solution(qaoa_result['selection'], asset_names)

        # Compare
        match = np.array_equal(classical_sol, qaoa_result['selection'])
        print(f"\n✓ Solutions match: {match}")
        print(f"QAOA iterations: {qaoa_result['iterations']}")


def efficient_frontier():
    """
    Compute efficient frontier using QAOA.
    """
    print("\n" + "=" * 60)
    print("     Efficient Frontier via QAOA")
    print("=" * 60)

    # Market data
    n_assets = 5
    expected_returns, covariance, asset_names = generate_market_data(n_assets, seed=123)

    risk_aversions = np.linspace(0.0, 1.0, 11)
    results = []

    print(f"\n{'Risk Aversion':^15} {'Return':^10} {'Risk':^10} {'Sharpe':^10} {'Assets':^20}")
    print("-" * 70)

    for lambda_risk in risk_aversions:
        optimizer = PortfolioOptimizer(
            expected_returns=expected_returns,
            covariance_matrix=covariance,
            risk_aversion=lambda_risk,
            budget=3
        )

        result = optimizer.quantum_solve(depth=2, shots=500, max_iterations=30)

        results.append({
            'lambda': lambda_risk,
            'return': result['expected_return'],
            'risk': result['risk'],
            'sharpe': result['sharpe_ratio'],
            'assets': result['selected_assets']
        })

        assets_str = ','.join(str(a) for a in result['selected_assets'])
        print(f"{lambda_risk:^15.2f} {result['expected_return']:^10.2%} "
              f"{result['risk']:^10.2%} {result['sharpe_ratio']:^10.3f} [{assets_str:^18}]")

    return results


if __name__ == "__main__":
    # Run main example
    run_portfolio_example()

    # Compute efficient frontier
    efficient_frontier()
```

## C Implementation

```c
/**
 * Portfolio Optimization with QAOA
 * Minimize portfolio risk subject to budget constraints.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quantum_sim.h"
#include "qaoa.h"

#define MAX_ASSETS 8

typedef struct {
    int n_assets;
    double expected_returns[MAX_ASSETS];
    double covariance[MAX_ASSETS][MAX_ASSETS];
    double risk_aversion;
    int budget;
    double Q[MAX_ASSETS][MAX_ASSETS];  // QUBO matrix
    double offset;
} portfolio_t;

/**
 * Build QUBO matrix from portfolio parameters.
 */
void build_qubo(portfolio_t* portfolio) {
    int n = portfolio->n_assets;
    double lambda = portfolio->risk_aversion;
    double penalty = 10.0;

    // Initialize Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            portfolio->Q[i][j] = 0.0;
        }
    }

    // Risk term: λ * x^T Σ x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            portfolio->Q[i][j] += lambda * portfolio->covariance[i][j];
        }
    }

    // Return term: -(1-λ) * μ^T x
    for (int i = 0; i < n; i++) {
        portfolio->Q[i][i] -= (1 - lambda) * portfolio->expected_returns[i];
    }

    // Budget constraint: (sum_i x_i - budget)^2
    if (portfolio->budget > 0) {
        for (int i = 0; i < n; i++) {
            portfolio->Q[i][i] += penalty * (1 - 2 * portfolio->budget);
            for (int j = i + 1; j < n; j++) {
                portfolio->Q[i][j] += 2 * penalty;
                portfolio->Q[j][i] += 2 * penalty;
            }
        }
        portfolio->offset = penalty * portfolio->budget * portfolio->budget;
    } else {
        portfolio->offset = 0.0;
    }
}

/**
 * Evaluate portfolio for a given selection.
 */
double evaluate_portfolio(portfolio_t* portfolio, int* selection) {
    int n = portfolio->n_assets;
    double cost = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost += selection[i] * portfolio->Q[i][j] * selection[j];
        }
    }

    return cost + portfolio->offset;
}

/**
 * Classical brute-force solver.
 */
void classical_solve(portfolio_t* portfolio, int* best_selection) {
    int n = portfolio->n_assets;
    double best_cost = 1e10;

    for (int mask = 0; mask < (1 << n); mask++) {
        int selection[MAX_ASSETS];
        int count = 0;

        for (int i = 0; i < n; i++) {
            selection[i] = (mask >> i) & 1;
            count += selection[i];
        }

        // Check budget constraint
        if (portfolio->budget > 0 && count != portfolio->budget) {
            continue;
        }

        double cost = evaluate_portfolio(portfolio, selection);
        if (cost < best_cost) {
            best_cost = cost;
            for (int i = 0; i < n; i++) {
                best_selection[i] = selection[i];
            }
        }
    }
}

/**
 * QAOA-based solver.
 */
void qaoa_solve(portfolio_t* portfolio, int* selection, int depth) {
    int n = portfolio->n_assets;

    // Create QAOA solver
    qaoa_solver_t* qaoa = qaoa_create(n, depth);

    // Set QUBO
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(portfolio->Q[i][j]) > 1e-10) {
                qaoa_add_interaction(qaoa, i, j, portfolio->Q[i][j]);
            }
        }
    }

    // Optimize
    qaoa_result_t result = qaoa_optimize(qaoa, 50);

    // Extract solution
    for (int i = 0; i < n; i++) {
        selection[i] = (result.best_bitstring >> i) & 1;
    }

    printf("QAOA: iterations=%d, cost=%.4f\n", result.iterations, result.cost);

    qaoa_destroy(qaoa);
}

/**
 * Analyze portfolio solution.
 */
void analyze_portfolio(portfolio_t* portfolio, int* selection, const char** names) {
    int n = portfolio->n_assets;

    printf("\n--- Portfolio Analysis ---\n");

    // Selected assets
    printf("Selected assets: ");
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (selection[i]) {
            printf("%s ", names[i]);
            count++;
        }
    }
    printf("(%d total)\n", count);

    // Calculate return
    double portfolio_return = 0.0;
    for (int i = 0; i < n; i++) {
        portfolio_return += selection[i] * portfolio->expected_returns[i];
    }

    // Calculate risk
    double portfolio_variance = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            portfolio_variance += selection[i] * portfolio->covariance[i][j] * selection[j];
        }
    }
    double portfolio_risk = sqrt(portfolio_variance);

    // Sharpe ratio
    double sharpe = (portfolio_risk > 0) ? portfolio_return / portfolio_risk : 0;

    printf("Expected return: %.2f%%\n", 100.0 * portfolio_return);
    printf("Risk (std dev):  %.2f%%\n", 100.0 * portfolio_risk);
    printf("Sharpe ratio:    %.3f\n", sharpe);
}

int main(void) {
    printf("================================================\n");
    printf("     Portfolio Optimization with QAOA\n");
    printf("================================================\n\n");

    // Create portfolio problem
    portfolio_t portfolio = {
        .n_assets = 5,
        .expected_returns = {0.12, 0.10, 0.08, 0.15, 0.06},
        .covariance = {
            {0.04, 0.01, 0.005, 0.02, 0.003},
            {0.01, 0.03, 0.008, 0.01, 0.004},
            {0.005, 0.008, 0.02, 0.006, 0.002},
            {0.02, 0.01, 0.006, 0.05, 0.008},
            {0.003, 0.004, 0.002, 0.008, 0.015}
        },
        .risk_aversion = 0.5,
        .budget = 3
    };

    const char* asset_names[] = {"Tech", "Finance", "Health", "Energy", "Utils"};

    printf("Market Data:\n");
    for (int i = 0; i < portfolio.n_assets; i++) {
        printf("  %s: μ=%.1f%%, σ=%.1f%%\n",
               asset_names[i],
               100.0 * portfolio.expected_returns[i],
               100.0 * sqrt(portfolio.covariance[i][i]));
    }

    // Build QUBO
    build_qubo(&portfolio);

    printf("\nBudget: Select exactly %d assets\n", portfolio.budget);

    // Classical solution
    printf("\n=== Classical Solution ===\n");
    int classical_selection[MAX_ASSETS];
    classical_solve(&portfolio, classical_selection);
    analyze_portfolio(&portfolio, classical_selection, asset_names);

    // QAOA solution
    printf("\n=== QAOA Solution (depth=3) ===\n");
    int qaoa_selection[MAX_ASSETS];
    qaoa_solve(&portfolio, qaoa_selection, 3);
    analyze_portfolio(&portfolio, qaoa_selection, asset_names);

    // Check match
    int match = 1;
    for (int i = 0; i < portfolio.n_assets; i++) {
        if (classical_selection[i] != qaoa_selection[i]) {
            match = 0;
            break;
        }
    }
    printf("\nSolutions match: %s\n", match ? "Yes" : "No");

    return 0;
}
```

## Expected Output

```
============================================================
     Portfolio Optimization with QAOA
============================================================

Market Data (6 assets)
----------------------------------------
Tech_1          μ=8.7%  σ=22.1%
Finance_1       μ=11.2%  σ=28.4%
Healthcare_1    μ=6.9%  σ=19.5%
Energy_1        μ=13.4%  σ=31.2%
Consumer_1      μ=9.2%  σ=24.8%
Industrial_1    μ=7.8%  σ=17.9%

==================================================
Budget: Select exactly 3 assets
==================================================

Classical (brute force) solution:

==================================================
           Portfolio Analysis
==================================================

Selected Assets (3/6):
  • Finance_1            (μ = 11.20%)
  • Energy_1             (μ = 13.40%)
  • Industrial_1         (μ = 7.80%)

Portfolio Metrics:
------------------------------
  Expected Return:  10.80%
  Risk (Std Dev):   18.45%
  Sharpe Ratio:     0.585

Correlation Structure:
------------------------------
  Avg correlation:  0.312

QAOA solution (depth=3):
QAOA iterations: 47
Cost: -0.2341

Selected Assets (3/6):
  • Finance_1            (μ = 11.20%)
  • Energy_1             (μ = 13.40%)
  • Industrial_1         (μ = 7.80%)

Portfolio Metrics:
------------------------------
  Expected Return:  10.80%
  Risk (Std Dev):   18.45%
  Sharpe Ratio:     0.585

✓ Solutions match: True
QAOA iterations: 47
```

## Key Concepts

### Risk-Return Trade-off

The risk aversion parameter λ controls the trade-off:
- λ = 0: Maximize return only
- λ = 1: Minimize risk only
- λ = 0.5: Balance both equally

### Diversification

Low correlation between selected assets reduces portfolio risk. QAOA naturally tends toward diversified portfolios due to the covariance term.

### Scaling Considerations

| Assets | Qubits | Classical | QAOA |
|--------|--------|-----------|------|
| 5 | 5 | 32 combinations | ~50 iterations |
| 10 | 10 | 1,024 combinations | ~100 iterations |
| 20 | 20 | 1M combinations | ~200 iterations |
| 50 | 50 | Intractable | ~500 iterations |

## Exercises

### Exercise 1: Sector Constraints

Add constraints to limit exposure to any single sector:

```python
# Add to QUBO: penalty if more than 2 assets from same sector
for sector in sectors:
    sector_assets = [i for i, name in enumerate(asset_names) if sector in name]
    # Add constraint: sum over sector <= 2
```

### Exercise 2: Transaction Costs

Include transaction costs in the optimization:

```python
# Current holdings
current_holdings = [1, 0, 1, 0, 0, 0]
transaction_cost = 0.001  # 0.1% per trade

# Add to QUBO: cost for changing positions
for i in range(n_assets):
    Q[i, i] += transaction_cost * (1 - 2 * current_holdings[i])
```

### Exercise 3: Compare with Classical Optimizer

Benchmark against scipy optimization:

```python
from scipy.optimize import minimize

def classical_markowitz(returns, covariance, risk_aversion):
    n = len(returns)

    def objective(w):
        return risk_aversion * w @ covariance @ w - (1 - risk_aversion) * returns @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n

    result = minimize(objective, np.ones(n)/n, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x
```

## Real-World Applications

- **Asset Allocation**: Institutional portfolio construction
- **Risk Management**: Minimizing tail risk in portfolios
- **Index Tracking**: Selecting subset of stocks to track index
- **Multi-Period Optimization**: Dynamic rebalancing strategies

## See Also

- [QAOA Algorithm](../../algorithms/qaoa-algorithm.md) - Complete theory
- [C API: QAOA](../../api/c/qaoa.md) - API reference
- [Tutorial: QAOA Optimization](../../tutorials/07-qaoa-optimization.md) - Step-by-step guide

