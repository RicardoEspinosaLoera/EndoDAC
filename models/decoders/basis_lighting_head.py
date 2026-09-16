"""Illumination calibration as a low-order polynomial field (--illum_calib basis).

`LightingDecoder` predicts one (c, b) pair per pixel and then blurs the result, which leaves
thousands of effective degrees of freedom. Measurements on this codebase (CVIU_REVISION_PLAN.md
sections 8e and 8g) show that capacity is spent absorbing photometric residual that is *not*
illumination -- most plausibly geometric error -- which removes from the photometric loss exactly
the signal the depth network needs. Two symptoms: the learned maps do not track an injected
illumination gain (slope -0.006), and the dense model gives worse depth than a two-parameter
global one while explaining ten times more photometric residual.

This head keeps the affine model of Eq. (1) but bounds *how local* it can be. The network predicts
2k coefficients and the field is their combination with a fixed polynomial basis over normalised
image coordinates u, v in [-1, 1]:

    c(u, v) = 1 + alpha * tanh( sum_j  w_j  phi_j(u, v) )
    b(u, v) =     beta  * tanh( sum_j  v_j  phi_j(u, v) )

with phi_j the monomials u^a v^b of total degree <= `degree`, so k = (d+1)(d+2)/2:

    degree 0 -> k = 1   a constant field, i.e. the global affine model of Ozyoruk et al.
    degree 1 -> k = 3   a linear gradient, what an off-axis light source produces
    degree 2 -> k = 6   quadratic: vignetting and radial fall-off
    degree 3 -> k = 10

Because the field cannot vary faster than the basis, it cannot follow depth edges (where geometric
error concentrates) and it cannot match the per-window mean and variance that SSIM normalises away
(window 7 in this repo). The calibration therefore explains regional illumination and leaves the
structured residual in the loss, where it still pushes the depth network.

The output keys and the alpha/beta bounds are those of LightingDecoder and GlobalLightingHead, so
the trainer, the evaluation stages and `calibration_supervision_loss` need no change. The field is
built at the image resolution the trainer passes in `grid`. An earlier version built it on a coarse
64x80 grid and let the trainer upsample: that is wrong, because tanh of a large polynomial is close
to a step and bilinear upsampling of the *post-tanh* field was off by 6% of alpha. The polynomial
itself interpolates almost exactly; its saturated image does not.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisLightingHead(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), degree=2, alpha=0.10, beta=0.05,
                 grid=(256, 320)):
        super(BasisLightingHead, self).__init__()
        self.scales = list(scales)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.beta = float(beta)

        basis = self._make_basis(self.degree, grid)          # (k, Hg, Wg)
        # persistent=False: the basis is a constant of the parameterisation, not something to
        # learn or to ship inside lighting.pth, and keeping it out means a checkpoint stays
        # loadable if the evaluation grid ever differs from the training one
        self.register_buffer("basis", basis, persistent=False)
        self.num_terms = basis.shape[0]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.head = nn.Conv2d(256, 2 * self.num_terms, 1)
        # start at the identity calibration (c=1, b=0); tanh'(0)=1 so gradients still flow
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _make_basis(degree, grid):
        """Monomials u^(d-i) v^i of total degree <= `degree`, on a normalised [-1, 1] grid."""
        h, w = grid
        v = torch.linspace(-1.0, 1.0, h).view(h, 1).expand(h, w)
        u = torch.linspace(-1.0, 1.0, w).view(1, w).expand(h, w)
        terms = []
        for d in range(degree + 1):
            for i in range(d + 1):
                terms.append((u ** (d - i)) * (v ** i))
        return torch.stack(terms, 0).contiguous()

    def forward(self, input_features):
        x = self.head(F.relu(self.squeeze(self.pool(input_features[-1]))))   # (B, 2k, 1, 1)
        coeff = x.view(x.shape[0], 2, self.num_terms)
        field = torch.einsum("bnk,khw->bnhw", coeff, self.basis)             # (B, 2, Hg, Wg)

        c = 1.0 + self.alpha * torch.tanh(field[:, 0:1])
        b = self.beta * torch.tanh(field[:, 1:2])

        outputs = {}
        for s in self.scales:
            outputs[("lighting", s)] = field
            outputs[("contrast", s)] = c
            outputs[("brightness", s)] = b
        return outputs
