import torch
from torch.autograd import Function

from fdint_core import (
    BackendOps,
    build_fermi_dirac_integral_minus_half,
    build_fermi_dirac_integral_half,
    build_fermi_dirac_integral_three_half,
    build_inverse_fermi_dirac_integral_half,
)

def _torch_select(conditions, results, default):
    out = default
    for cond, value in zip(reversed(conditions), reversed(results)):
        out = torch.where(cond, value, out)
    return out


def _torch_maximum(x, value):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    else:
        value = value.to(dtype=x.dtype, device=x.device)
    return torch.maximum(x, value)


def _torch_tiny(x):
    tensor = x if torch.is_tensor(x) else torch.as_tensor(x)
    return torch.finfo(tensor.dtype).tiny


_TORCH_BACKEND = BackendOps(
    exp=torch.exp,
    sqrt=torch.sqrt,
    power=torch.pow,
    maximum=_torch_maximum,
    select=_torch_select,
    asarray=torch.as_tensor,
    log=torch.log,
    tiny=_torch_tiny,
)


fermi_dirac_integral_minus_half = build_fermi_dirac_integral_minus_half(_TORCH_BACKEND)
_fermi_dirac_integral_half_impl = build_fermi_dirac_integral_half(_TORCH_BACKEND)
_fermi_dirac_integral_three_half_impl = build_fermi_dirac_integral_three_half(_TORCH_BACKEND)
_inverse_fdi_half_impl = build_inverse_fermi_dirac_integral_half(_TORCH_BACKEND)


class InverseFermiDiracIntegralHalf(Function):
    @staticmethod
    def forward(ctx, x):
        y = _inverse_fdi_half_impl(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad_input = grad_output / fermi_dirac_integral_minus_half(y)
        return grad_input


inverse_fermi_dirac_integral_half = InverseFermiDiracIntegralHalf.apply


class FermiDiracIntegralHalf(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _fermi_dirac_integral_half_impl(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output * fermi_dirac_integral_minus_half(x)
        return grad_input


fermi_dirac_integral_half = FermiDiracIntegralHalf.apply


class FermiDiracIntegralThreeHalf(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _fermi_dirac_integral_three_half_impl(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output * fermi_dirac_integral_half(x)
        return grad_input


fermi_dirac_integral_three_half = FermiDiracIntegralThreeHalf.apply




