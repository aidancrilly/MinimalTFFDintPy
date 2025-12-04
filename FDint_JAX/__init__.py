import jax
import jax.numpy as jnp

from fdint_core import (
    BackendOps,
    build_fermi_dirac_integral_minus_half,
    build_fermi_dirac_integral_half,
    build_fermi_dirac_integral_three_half,
    build_inverse_fermi_dirac_integral_half,
)

def _jax_select(conditions, results, default):
    return jnp.select(conditions, results, default)


def _jax_tiny(x):
    dtype = jnp.asarray(x).dtype
    return jnp.finfo(dtype).tiny


_JAX_BACKEND = BackendOps(
    exp=jnp.exp,
    sqrt=jnp.sqrt,
    power=jnp.power,
    maximum=jnp.maximum,
    select=_jax_select,
    asarray=jnp.asarray,
    log=jnp.log,
    tiny=_jax_tiny,
)


fermi_dirac_integral_minus_half = build_fermi_dirac_integral_minus_half(_JAX_BACKEND)
_fermi_dirac_integral_half_impl = build_fermi_dirac_integral_half(_JAX_BACKEND)
_fermi_dirac_integral_three_half_impl = build_fermi_dirac_integral_three_half(_JAX_BACKEND)
_inverse_fdi_half_impl = build_inverse_fermi_dirac_integral_half(_JAX_BACKEND)


@jax.custom_vjp
def inverse_fermi_dirac_integral_half(x):
    return _inverse_fdi_half_impl(x)


def inverse_fd_half_fwd(x):
    y = _inverse_fdi_half_impl(x)
    return y, y


def inverse_fd_half_bwd(res, g):
    return (g / fermi_dirac_integral_minus_half(res),)


inverse_fermi_dirac_integral_half.defvjp(inverse_fd_half_fwd, inverse_fd_half_bwd)


@jax.custom_vjp
def fermi_dirac_integral_half(x):
    return _fermi_dirac_integral_half_impl(x)


def fermi_dirac_integral_half_fwd(x):
    y = _fermi_dirac_integral_half_impl(x)
    return y, fermi_dirac_integral_minus_half(x)


def FD_half_bwd(res, g):
    return (res * g,)


fermi_dirac_integral_half.defvjp(fermi_dirac_integral_half_fwd, FD_half_bwd)


@jax.custom_vjp
def fermi_dirac_integral_three_half(x):
    return _fermi_dirac_integral_three_half_impl(x)


def fermi_dirac_integral_three_half_fwd(x):
    y = _fermi_dirac_integral_three_half_impl(x)
    return y, fermi_dirac_integral_half(x)


def FD_three_halfs_bwd(res, g):
    return (res * g,)


fermi_dirac_integral_three_half.defvjp(
    fermi_dirac_integral_three_half_fwd, FD_three_halfs_bwd
)




