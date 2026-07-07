import jax.numpy as jnp
import numpy as np
import jax

from FDint_JAX import (
    fermi_dirac_integral_minus_half,
    fermi_dirac_integral_half,
    fermi_dirac_integral_three_half,
    inverse_fermi_dirac_integral_half,
)

def test_fermi_dirac_integrals():
    x_ref, ifd1h_ref, fd1h_ref, fd3h_ref = np.loadtxt(
        "tests/FDINT_values.csv", unpack=True, delimiter=","
    )

    x = jnp.asarray(x_ref) # Use x_ref directly as jax array

    fdm1h = fermi_dirac_integral_minus_half(x)
    fd1h = fermi_dirac_integral_half(x)
    fd3h = fermi_dirac_integral_three_half(x)
    ifd1h = inverse_fermi_dirac_integral_half(x)

    np.testing.assert_allclose(fd1h, fd1h_ref, rtol=1e-5)
    np.testing.assert_allclose(fd3h, fd3h_ref, rtol=1e-5)
    np.testing.assert_allclose(ifd1h, ifd1h_ref, rtol=1e-5)
    np.testing.assert_allclose(fermi_dirac_integral_half(ifd1h), x_ref, rtol=1e-5)

    AD_fdm1h = jax.vmap(jax.grad(fermi_dirac_integral_half))(x)
    AD_fd1h = jax.vmap(jax.grad(fermi_dirac_integral_three_half))(x)
    AD_ifd1h = jax.vmap(jax.grad(inverse_fermi_dirac_integral_half))(x)

    np.testing.assert_allclose(AD_fdm1h, fdm1h, rtol=1e-5)
    np.testing.assert_allclose(AD_fd1h, fd1h, rtol=1e-5)
    np.testing.assert_allclose(
        AD_ifd1h, 1.0 / fermi_dirac_integral_minus_half(ifd1h), rtol=1e-5
    )


def test_fermi_dirac_integrals_negative_x():
    # The main reference grid is positive-only, which previously hid wrong
    # coefficients in the x < 0 regions of F_{-1/2}. This locks in negative-x
    # accuracy against high-precision mpmath values.
    x_ref, fdm1h_ref, fd1h_ref, fd3h_ref = np.loadtxt(
        "tests/FDINT_negative_values.csv", unpack=True, delimiter=","
    )

    x = jnp.asarray(x_ref)

    np.testing.assert_allclose(
        fermi_dirac_integral_minus_half(x), fdm1h_ref, rtol=1e-5
    )
    np.testing.assert_allclose(fermi_dirac_integral_half(x), fd1h_ref, rtol=1e-5)
    np.testing.assert_allclose(
        fermi_dirac_integral_three_half(x), fd3h_ref, rtol=1e-5
    )

