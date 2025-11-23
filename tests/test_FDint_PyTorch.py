import torch
import numpy as np

from FDint_PyTorch import fermi_dirac_integral_minus_half, fermi_dirac_integral_half, fermi_dirac_integral_three_half

def test_fermi_dirac_integrals():
    x_ref, _, fd1h_ref, fd3h_ref = np.loadtxt(
        "tests/FDINT_values.csv", unpack=True, delimiter=","
    )

    x = torch.tensor(x_ref) # Use x_ref directly as jax array

    fdm1h = fermi_dirac_integral_minus_half(x)
    fd1h = fermi_dirac_integral_half(x)
    fd3h = fermi_dirac_integral_three_half(x)

    np.testing.assert_allclose(fd1h, fd1h_ref, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(fd3h, fd3h_ref, rtol=1e-5, atol=1e-8)

    x.requires_grad_(True)
    y1 = fermi_dirac_integral_half(x)
    y1.sum().backward()
    AD_fdm1h = x.grad.clone()
    x.grad.zero_()
    y2 = fermi_dirac_integral_three_half(x)
    y2.sum().backward()
    AD_fd1h = x.grad.clone()
    x.requires_grad_(False)

    np.testing.assert_allclose(AD_fdm1h, fdm1h, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(AD_fd1h, fd1h, rtol=1e-5, atol=1e-8)