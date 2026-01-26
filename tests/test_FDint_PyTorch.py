import torch
import numpy as np

from FDint_PyTorch import (
    fermi_dirac_integral_minus_half,
    fermi_dirac_integral_half,
    fermi_dirac_integral_three_half,
    inverse_fermi_dirac_integral_half,
)

def test_fermi_dirac_integrals():
    x_ref, ifd1h_ref, fd1h_ref, fd3h_ref = np.loadtxt(
        "tests/FDINT_values.csv", unpack=True, delimiter=","
    )

    x = torch.tensor(x_ref) # Use x_ref directly as jax array

    fdm1h = fermi_dirac_integral_minus_half(x)
    fd1h = fermi_dirac_integral_half(x)
    fd3h = fermi_dirac_integral_three_half(x)
    ifd1h = inverse_fermi_dirac_integral_half(x)

    np.testing.assert_allclose(fd1h, fd1h_ref, rtol=1e-5)
    np.testing.assert_allclose(fd3h, fd3h_ref, rtol=1e-5)
    np.testing.assert_allclose(ifd1h.detach().numpy(), ifd1h_ref, rtol=1e-5)
    np.testing.assert_allclose(
        fermi_dirac_integral_half(ifd1h).detach().numpy(), x_ref, rtol=1e-5
    )

    x.requires_grad_(True)
    y1 = fermi_dirac_integral_half(x)
    y1.sum().backward()
    AD_fdm1h = x.grad.clone()
    x.grad.zero_()
    y2 = fermi_dirac_integral_three_half(x)
    y2.sum().backward()
    AD_fd1h = x.grad.clone()
    x.requires_grad_(False)

    x_inv = torch.tensor(x_ref, dtype=x.dtype)
    x_inv.requires_grad_(True)
    inv_vals = inverse_fermi_dirac_integral_half(x_inv)
    inv_vals.sum().backward()
    AD_ifd1h = x_inv.grad.clone()
    x_inv.grad.zero_()
    x_inv.requires_grad_(False)

    np.testing.assert_allclose(AD_fdm1h, fdm1h, rtol=1e-5)
    np.testing.assert_allclose(AD_fd1h, fd1h, rtol=1e-5)
    expected_grad = 1.0 / fermi_dirac_integral_minus_half(inv_vals.detach())
    np.testing.assert_allclose(
        AD_ifd1h.detach().numpy(), expected_grad.detach().numpy(), rtol=1e-5
    )
