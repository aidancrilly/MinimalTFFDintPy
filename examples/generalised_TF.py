import numpy as np
from scipy.integrate import solve_bvp
from FDint_JAX import fermi_dirac_integral_half

def I(x):
    return fermi_dirac_integral_half(x)

def solve_beta_bvp(a, b, *, n_points=200, eps=1e-8, max_nodes=20000, tol=1e-6):
    """
    Solve:
        d/dw ( (beta'(w))/w ) = 2 w^3 I( 2 beta(w) / w^2 )
    with BCs:
        beta(0) = a
        beta'(b) = 2 beta(b) / b

    Uses solve_bvp on [eps, b] to avoid division by zero at w=0.
    """

    # ---- ODE as first-order system ----
    # Let y0 = beta, y1 = beta'
    #
    # Given: d/dw (y1 / w) = 2 w^3 I( 2 y0 / w^2 )
    # Expand: (y1/w)' = y1'/w - y1/w^2
    # So: y1'/w - y1/w^2 = 2 w^3 I(...)
    # => y1' = y1/w + 2 w^4 I(...)
    def fun(w, y):
        beta = y[0]
        dbeta = y[1]

        # avoid division issues
        w_safe = np.maximum(w, eps)

        arg = 2.0 * beta / (w_safe**2)
        rhs = 2.0 * (w_safe**4) * I(arg)

        d_beta = dbeta
        d_dbeta = (dbeta / w_safe) + rhs
        return np.vstack((d_beta, d_dbeta))

    # ---- Boundary conditions ----
    # At w=0: beta(0)=a  -> enforce at w=eps: beta(eps)=a
    # At w=b: beta'(b) - 2 beta(b)/b = 0
    def bc(ya, yb):
        return np.array([
            ya[0] - a,
            yb[1] - 2.0 * yb[0] / b
        ])

    # ---- Mesh & initial guess ----
    w = np.linspace(eps, b, n_points)

    # A reasonable guess that satisfies beta(eps)=a and roughly respects the slope condition:
    # try beta ~ a*(w/eps)^2 near 0 is too aggressive; instead use a gentle quadratic anchored at a.
    # We'll pick beta ~ a, dbeta ~ (2a/b) w as a mild slope.
    beta_guess = a * np.ones_like(w)
    dbeta_guess = (2.0 * a / b) * (w / b)
    y_guess = np.vstack((beta_guess, dbeta_guess))

    sol = solve_bvp(fun, bc, w, y_guess, tol=tol, max_nodes=max_nodes)

    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    return sol

# ---- Example usage ----
if __name__ == "__main__":
    a = 100.0
    b = 0.5

    sol = solve_beta_bvp(a, b, eps = 1e-4, max_nodes=int(1e5), tol=1e-3)

    # Evaluate solution on a fine grid
    w_plot = np.linspace(1e-8, b, 400)
    beta = sol.sol(w_plot)[0]
    dbeta = sol.sol(w_plot)[1]

    import matplotlib.pyplot as plt

    plt.plot(w_plot,beta)
    plt.show()