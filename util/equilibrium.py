"""
Equilibrium handling: load a G-EQDSK file via `freeqdsk` and compute the
geometric and electromagnetic quantities the rest of the simulator needs.

What this module gives you
--------------------------
From a g-file alone:
  - Plasma volume V [m^3], surface area S_lcfs [m^2], cross-section area A [m^2]
  - Major radius R0, minor radius a, elongation kappa, triangularity delta
  - Magnetic axis (R_axis, Z_axis), B_t at R0
  - Poloidal field B_p(R,Z) and |B_p|^2 volume integral
  - Internal inductance l_i (ITER definition, l_i(3))
  - Plasma self-inductance L_p [H], by both formula and integral
  - Plasma-current consistency check via Ampere's law on LCFS
  - q-profile, F(psi), p(psi), psi(R,Z)

Conventions
-----------
We use SI units. Sign of psi follows freeqdsk's cocos kwarg; for everything we
compute downstream only |B_p|^2 matters so the sign is harmless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline
from freeqdsk import geqdsk

MU0 = 4.0e-7 * np.pi


# ---------- container ------------------------------------------------------


@dataclass
class Equilibrium:
    """Parsed g-file + derived geometric/magnetic quantities."""

    # Raw fields (subset we use)
    R: np.ndarray  # 1D, [nR]
    Z: np.ndarray  # 1D, [nZ]
    psi_RZ: np.ndarray  # 2D, [nR, nZ]   psi on (R,Z) grid
    psi_axis: float
    psi_bdry: float
    R_axis: float
    Z_axis: float
    Bt_at_Rref: float  # vacuum toroidal field at R_ref
    R_ref: float  # reference major radius (rcentr)
    Ip: float  # plasma current [A]
    R_bdry: np.ndarray  # LCFS R points
    Z_bdry: np.ndarray  # LCFS Z points
    psi_n: np.ndarray  # normalized psi on 1D profile grid
    fpol: np.ndarray  # F(psi) = R B_t
    pressure: np.ndarray  # p(psi)
    qpsi: np.ndarray  # q(psi)
    # Source/meta
    source_file: Optional[str] = None
    cocos: int = 1

    # Derived geometry (filled by compute_geometry)
    R0: float = field(default=np.nan)  # geometric major radius
    a: float = field(default=np.nan)  # minor radius
    kappa: float = field(default=np.nan)  # elongation
    delta_upper: float = field(default=np.nan)
    delta_lower: float = field(default=np.nan)
    delta: float = field(default=np.nan)  # average triangularity
    V: float = field(default=np.nan)  # plasma volume [m^3]
    S_lcfs: float = field(default=np.nan)  # LCFS surface area [m^2]
    A_cs: float = field(default=np.nan)  # poloidal cross-section [m^2]
    # Derived magnetics
    Bt0: float = field(default=np.nan)  # B_t at R0
    li3: float = field(default=np.nan)  # internal inductance, l_i(3)
    Lp_internal_H: float = field(
        default=np.nan
    )  # internal L_p from |B_p|^2 inside LCFS
    Lp_total_grid_H: float = field(
        default=np.nan
    )  # total L_p from |B_p|^2 over full grid
    Lp_formula_H: float = field(default=np.nan)  # total L_p, Wesson elongated formula
    Ip_check_A: float = field(default=np.nan)  # Ampere's-law sanity check on LCFS

    # Spline of psi for fast (R,Z) evaluation
    _psi_spline: Optional[RectBivariateSpline] = field(default=None, repr=False)


# ---------- loader ---------------------------------------------------------


def load_gfile(path: str | Path, cocos: int = 1) -> Equilibrium:
    """Read a G-EQDSK file with freeqdsk and return a populated Equilibrium."""
    path = Path(path)
    with open(path, "r") as fh:
        g = geqdsk.read(fh, cocos=cocos)

    # Build (R,Z) grid. freeqdsk stores psi as psi[nR, nZ] with R along axis 0.
    R = g.rleft + np.linspace(0.0, g.rdim, g.nx)
    Z = g.zmid - 0.5 * g.zdim + np.linspace(0.0, g.zdim, g.ny)

    psi_RZ = np.asarray(g.psi)  # shape (nR, nZ)
    psi_n = np.linspace(0.0, 1.0, len(g.fpol))

    eq = Equilibrium(
        R=R,
        Z=Z,
        psi_RZ=psi_RZ,
        psi_axis=float(g.simagx),
        psi_bdry=float(g.sibdry),
        R_axis=float(g.rmagx),
        Z_axis=float(g.zmagx),
        Bt_at_Rref=float(g.bcentr),
        R_ref=float(g.rcentr),
        Ip=float(g.cpasma),
        R_bdry=np.asarray(g.rbdry),
        Z_bdry=np.asarray(g.zbdry),
        psi_n=psi_n,
        fpol=np.asarray(g.fpol),
        pressure=np.asarray(g.pres),
        qpsi=np.asarray(g.qpsi),
        source_file=str(path),
        cocos=cocos,
    )
    eq._psi_spline = RectBivariateSpline(eq.R, eq.Z, eq.psi_RZ, kx=3, ky=3)
    compute_geometry(eq)
    compute_inductance(eq)
    return eq


# ---------- geometry -------------------------------------------------------


def _ensure_closed(R: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Append first point if contour isn't closed."""
    if R[0] != R[-1] or Z[0] != Z[-1]:
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
    return R, Z


def compute_geometry(eq: Equilibrium) -> None:
    """Geometric scalars from the LCFS contour. Sets fields on `eq` in place."""
    R, Z = _ensure_closed(eq.R_bdry, eq.Z_bdry)

    R_min, R_max = R.min(), R.max()
    Z_min, Z_max = Z.min(), Z.max()
    R0 = 0.5 * (R_max + R_min)
    a = 0.5 * (R_max - R_min)
    b = 0.5 * (Z_max - Z_min)
    kappa = b / a

    # Triangularity: R at extreme Z, referenced to R0 and normalized to a
    R_at_Zmax = R[np.argmax(Z)]
    R_at_Zmin = R[np.argmin(Z)]
    delta_u = (R0 - R_at_Zmax) / a
    delta_l = (R0 - R_at_Zmin) / a
    delta = 0.5 * (delta_u + delta_l)

    # Cross-section area by shoelace (signed; take |.|).
    # A = 1/2 |sum_i (R_i Z_{i+1} - R_{i+1} Z_i)|
    A_cs = 0.5 * np.abs(np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1]))

    # Volume of revolution (axisymmetric): V = pi * oint R^2 dZ.
    # Use trapezoidal integration around the closed contour.
    dZ = np.diff(Z)
    R_mid = 0.5 * (R[:-1] + R[1:])
    V = np.pi * np.abs(np.sum(R_mid**2 * dZ))

    # Surface area of revolution: S = 2*pi * oint R ds
    dR = np.diff(R)
    ds = np.sqrt(dR**2 + dZ**2)
    S = 2.0 * np.pi * np.sum(R_mid * ds)

    eq.R0 = R0
    eq.a = a
    eq.kappa = kappa
    eq.delta_upper = delta_u
    eq.delta_lower = delta_l
    eq.delta = delta
    eq.V = V
    eq.S_lcfs = S  # type: ignore
    eq.A_cs = A_cs

    # B_t at R0 (vacuum + plasma contribution is captured by F(psi) at boundary,
    # but for global scalars the vacuum value scaled to R0 is what people quote)
    eq.Bt0 = eq.Bt_at_Rref * eq.R_ref / R0


# ---------- magnetics ------------------------------------------------------


def B_poloidal_on_grid(eq: Equilibrium) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (B_R, B_Z, |B_p|) on the (R,Z) grid.

    Axisymmetric: B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR.

    Some g-files (especially spherical-tokamak and free-boundary) have grids
    that extend to or past R=0, which would produce inf in B and then NaN
    when multiplied by a mask that is False there. We guard against that by
    zeroing B at R<=0; those points are always outside the LCFS for any
    physical equilibrium, so any subsequent integral inside the LCFS is
    unaffected.
    """
    sp = eq._psi_spline
    assert sp is not None
    RR, _ = np.meshgrid(eq.R, eq.Z, indexing="ij")
    dpsi_dR = sp(eq.R, eq.Z, dx=1, dy=0)
    dpsi_dZ = sp(eq.R, eq.Z, dx=0, dy=1)

    # Safe division: produce 0 wherever R <= 0 (or numerically tiny).
    R_min_safe = 1e-12
    valid = RR > R_min_safe
    B_R = np.divide(-dpsi_dZ, RR, out=np.zeros_like(RR), where=valid)
    B_Z = np.divide(dpsi_dR, RR, out=np.zeros_like(RR), where=valid)
    B_p = np.sqrt(B_R**2 + B_Z**2)
    return B_R, B_Z, B_p


def _point_in_lcfs_mask(eq: Equilibrium) -> np.ndarray:
    """Boolean mask over (R,Z) grid: True if inside the LCFS contour."""
    from matplotlib.path import Path as MplPath

    R, Z = _ensure_closed(eq.R_bdry, eq.Z_bdry)
    poly = MplPath(np.column_stack([R, Z]))
    RR, ZZ = np.meshgrid(eq.R, eq.Z, indexing="ij")
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    return poly.contains_points(pts).reshape(RR.shape)


def compute_inductance(eq: Equilibrium) -> None:
    """Internal inductance l_i(3) and plasma self-inductance L_p.

    Definitions (ITER / Wesson conventions):
      l_i(3) = (2 * V * <B_p^2>) / (mu0^2 * Ip^2 * R0)
      L_p_internal = mu0 * R0 * l_i(3) / 2
                   = (1/mu0) * integral_{inside LCFS} B_p^2 dV / Ip^2
      L_p_total (grid integral) = (1/mu0) * integral_{full (R,Z) grid} B_p^2 dV / Ip^2
                  (informational; contaminated by external coil fields)
      L_p_total (formula, Wesson elongated approx):
            L_p ≈ mu0 * R0 * [ln(8 R0 / (a * sqrt(kappa))) + l_i/2 - 2]

    Recommendation: use `Lp_formula_H` as the plasma self-inductance in the
    circuit equation. The grid-integral is reported only as a sanity check
    on the magnitude.
    """
    _, _, B_p = B_poloidal_on_grid(eq)
    mask = _point_in_lcfs_mask(eq)

    # Volume element of a torus cell: dV = 2 pi R dR dZ
    dR = eq.R[1] - eq.R[0]
    dZ = eq.Z[1] - eq.Z[0]
    RR, _ = np.meshgrid(eq.R, eq.Z, indexing="ij")
    dV = 2.0 * np.pi * RR * dR * dZ

    Bp2_inside = np.sum((B_p**2) * dV * mask)  # ∫ B_p^2 dV inside LCFS
    Bp2_grid = np.sum((B_p**2) * dV)  # ∫ B_p^2 dV over full grid
    Bp2_avg_inside = Bp2_inside / eq.V if eq.V > 0 else np.nan

    eq.li3 = (2.0 * eq.V * Bp2_avg_inside) / (MU0**2 * eq.Ip**2 * eq.R0)

    # Internal L_p (just plasma volume integral)
    eq.Lp_internal_H = Bp2_inside / (MU0 * eq.Ip**2)
    # Total L_p (full-grid integral). Only meaningful if grid extends well beyond LCFS.
    eq.Lp_total_grid_H = Bp2_grid / (MU0 * eq.Ip**2)
    # Total L_p (Wesson elongated approx).
    eq.Lp_formula_H = (
        MU0
        * eq.R0
        * (
            np.log(8.0 * eq.R0 / (eq.a * np.sqrt(max(eq.kappa, 1e-6))))
            + 0.5 * eq.li3
            - 2.0
        )
    )

    # Ampere's-law check: Ip = (1/mu0) * oint B_pol · dl around the LCFS.
    # The sign of this integral depends on contour-traversal direction. We force
    # CCW traversal (positive signed area via the shoelace formula) so that the
    # sign of Ip_check_A matches the sign of a current flowing in +phi-hat by
    # the right-hand rule. After this, |sign(Ip_check_A) - sign(Ip)| signals
    # whether the g-file uses a COCOS convention where +Ip is opposite to ours.
    Rb, Zb = _ensure_closed(eq.R_bdry, eq.Z_bdry)
    signed_area = 0.5 * np.sum(Rb[:-1] * Zb[1:] - Rb[1:] * Zb[:-1])
    if signed_area < 0.0:  # contour is CW -> reverse
        Rb = Rb[::-1]
        Zb = Zb[::-1]

    dRb = np.diff(Rb)
    dZb = np.diff(Zb)
    Rm = 0.5 * (Rb[:-1] + Rb[1:])
    Zm = 0.5 * (Zb[:-1] + Zb[1:])
    sp = eq._psi_spline
    assert sp is not None
    dpsi_dR_b = sp(Rm, Zm, dx=1, dy=0, grid=False)
    dpsi_dZ_b = sp(Rm, Zm, dx=0, dy=1, grid=False)
    Br_b = -dpsi_dZ_b / Rm
    Bz_b = dpsi_dR_b / Rm
    circ_int = np.sum(Br_b * dRb + Bz_b * dZb)
    eq.Ip_check_A = circ_int / MU0  # type: ignore


# ---------- summary --------------------------------------------------------


def summary(eq: Equilibrium) -> str:
    # Signed comparison so a flip is visible.
    Ip_mag_err_pct = 100.0 * (abs(eq.Ip_check_A) - abs(eq.Ip)) / abs(eq.Ip)
    sign_flip = (eq.Ip_check_A * eq.Ip) < 0.0

    # COCOS sanity. If |Ip_check / Ip| is near 2*pi or 1/(2*pi), the user
    # almost certainly has the cocos kwarg wrong for their file.
    ratio = abs(eq.Ip_check_A) / max(abs(eq.Ip), 1e-30)
    cocos_msg = ""
    if 5.5 < ratio < 7.0:
        cocos_msg = (
            "    [!] |Ip_check/Ip| ~ 2*pi: g-file is likely COCOS 11+, "
            "but you loaded as COCOS <10. Try gfile_cocos: 11.\n"
        )
    elif 0.13 < ratio < 0.20:
        cocos_msg = (
            "    [!] |Ip_check/Ip| ~ 1/(2*pi): g-file is likely COCOS <10, "
            "but you loaded as COCOS 11+. Try gfile_cocos: 1.\n"
        )

    sign_msg = ""
    if sign_flip:
        sign_msg = (
            "    [i] Ip_check has opposite sign to Ip; g-file's COCOS "
            "uses the reverse +phi-hat convention. Magnitudes still "
            "agree, physics unaffected.\n"
        )

    return (
        f"Equilibrium summary  (source: {eq.source_file})\n"
        f"  Geometry:\n"
        f"    R0 = {eq.R0:.3f} m, a = {eq.a:.3f} m, kappa = {eq.kappa:.3f}, "
        f"delta = {eq.delta:.3f} (up {eq.delta_upper:+.3f}, lo {eq.delta_lower:+.3f})\n"
        f"    Volume V          = {eq.V:.3f} m^3\n"
        f"    LCFS surface S    = {eq.S_lcfs:.3f} m^2\n"
        f"    Cross-section A   = {eq.A_cs:.3f} m^2\n"
        f"  Field & current:\n"
        f"    Bt @ R0           = {eq.Bt0:.3f} T\n"
        f"    Ip (from g-file)  = {eq.Ip/1e6:+.4f} MA\n"
        f"    Ip (Ampere check) = {eq.Ip_check_A/1e6:+.4f} MA  "
        f"(|err| = {Ip_mag_err_pct:+.2f}%)\n"
        f"{cocos_msg}{sign_msg}"
        f"  Inductance:\n"
        f"    l_i(3)            = {eq.li3:.3f}\n"
        f"    L_p internal      = {eq.Lp_internal_H*1e6:.2f} uH   (inside LCFS only)\n"
        f"    L_p total (grid)  = {eq.Lp_total_grid_H*1e6:.2f} uH   (full (R,Z) grid)\n"
        f"    L_p Wesson fmla   = {eq.Lp_formula_H*1e6:.2f} uH   (total, elongated approx)\n"
    )
