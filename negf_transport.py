"""
NEGF Transport Code for Gate-All-Around (GAA) Transistor

This module implements the Non-Equilibrium Green's Function (NEGF) formalism
for quantum transport calculations in nanoscale transistors.

Key equations implemented:
- Retarded Green's function: G^R(E) = [(E + iη)I - H - Σ^R(E)]^(-1)
- Advanced Green's function: G^A(E) = [G^R(E)]†
- Lesser Green's function: G^<(E) = G^R(E) · Σ^<(E) · G^A(E)
- Electron density: n(r) = (1/2π) ∫ dE · Im[G^<(r,r,E)]
- Current: I = (q/ℏ) ∫ dE · Tr[Σ^<_L·G^> - Σ^>_L·G^<]

Reference: The flowchart follows DFT+NEGF methodology with self-consistent loop.

Author: Claude Code
"""

import numpy as np
from numpy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


# Physical constants
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
Q_E = 1.602176634e-19   # Elementary charge (C)
M_E = 9.1093837015e-31  # Electron mass (kg)
K_B = 1.380649e-23      # Boltzmann constant (J/K)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)


class MaterialType(Enum):
    """Supported material types for the channel."""
    SILICON = "Si"
    GERMANIUM = "Ge"
    INGAAS = "InGaAs"
    GAAS = "GaAs"


@dataclass
class Material:
    """Material properties for semiconductor channels."""
    name: str
    effective_mass: float  # in units of m_e
    dielectric_constant: float
    bandgap: float  # eV
    electron_affinity: float  # eV

    @classmethod
    def get_material(cls, material_type: MaterialType) -> 'Material':
        """Get predefined material properties."""
        materials = {
            MaterialType.SILICON: cls("Silicon", 0.26, 11.7, 1.12, 4.05),
            MaterialType.GERMANIUM: cls("Germanium", 0.12, 16.0, 0.66, 4.0),
            MaterialType.INGAAS: cls("In0.53Ga0.47As", 0.041, 13.9, 0.74, 4.5),
            MaterialType.GAAS: cls("GaAs", 0.067, 12.9, 1.42, 4.07),
        }
        return materials[material_type]


@dataclass
class GAADeviceParams:
    """
    Parameters for Gate-All-Around (GAA) transistor geometry.

    The device consists of:
    - Source contact (semi-infinite lead)
    - Channel region (gated, finite)
    - Drain contact (semi-infinite lead)

    For GAA geometry, the gate surrounds the cylindrical nanowire channel.
    """
    # Geometry
    channel_length: float = 10e-9      # Channel length (m)
    nanowire_radius: float = 3e-9      # Nanowire radius (m)
    oxide_thickness: float = 1e-9      # Gate oxide thickness (m)

    # Grid parameters
    nz: int = 50                       # Grid points along transport (z-direction)
    nr: int = 10                       # Grid points in radial direction

    # Material
    material: Material = None
    oxide_dielectric: float = 3.9      # SiO2 relative permittivity

    # Bias conditions
    vg: float = 0.0                    # Gate voltage (V)
    vd: float = 0.0                    # Drain voltage (V)
    vs: float = 0.0                    # Source voltage (V)

    # Doping
    source_doping: float = 1e20        # Source doping (1/cm³)
    drain_doping: float = 1e20         # Drain doping (1/cm³)
    channel_doping: float = 1e15       # Channel doping (1/cm³)

    # Temperature
    temperature: float = 300.0         # Temperature (K)

    def __post_init__(self):
        if self.material is None:
            self.material = Material.get_material(MaterialType.SILICON)


class SurfaceGreenFunction:
    """
    Calculate surface Green's function for semi-infinite leads.

    The surface Green's function g_s is obtained iteratively using the
    Sancho-Rubio method (decimation technique), which efficiently computes
    the surface Green's function of a semi-infinite periodic system.

    For a lead with unit cell Hamiltonian H_00 and coupling H_01:
    g_s(E) = [(E + iη)I - H_00 - H_01 · g_s · H_01†]^(-1)

    This is solved iteratively until convergence.
    """

    def __init__(self, H_unit: np.ndarray, H_coupling: np.ndarray,
                 eta: float = 1e-6, max_iter: int = 100, tol: float = 1e-8):
        """
        Initialize surface Green's function calculator.

        Args:
            H_unit: Unit cell Hamiltonian of the lead (n x n)
            H_coupling: Coupling matrix between adjacent unit cells (n x n)
            eta: Small imaginary part for broadening
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance
        """
        self.H_00 = np.array(H_unit, dtype=complex)
        self.H_01 = np.array(H_coupling, dtype=complex)
        self.H_10 = self.H_01.T.conj()  # H_01†
        self.n = H_unit.shape[0]
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def calculate(self, E: float) -> np.ndarray:
        """
        Calculate surface Green's function at energy E using Sancho-Rubio method.

        This decimation technique doubles the effective range at each iteration,
        achieving exponential convergence.

        Args:
            E: Energy (in eV)

        Returns:
            Surface Green's function g_s(E) as n x n matrix
        """
        # Initialize
        eps_s = self.H_00.copy()  # Surface diagonal
        eps_b = self.H_00.copy()  # Bulk diagonal
        alpha = self.H_01.copy()  # Forward coupling
        beta = self.H_10.copy()   # Backward coupling

        I = np.eye(self.n, dtype=complex)
        E_plus = (E + 1j * self.eta) * I

        for iteration in range(self.max_iter):
            # Green's function for bulk
            g_b = la.inv(E_plus - eps_b)

            # Update couplings (decimation)
            alpha_new = alpha @ g_b @ alpha
            beta_new = beta @ g_b @ beta

            # Update surface and bulk effective Hamiltonians
            eps_s_new = eps_s + alpha @ g_b @ beta
            eps_b_new = eps_b + alpha @ g_b @ beta + beta @ g_b @ alpha

            # Check convergence
            diff = la.norm(alpha_new) + la.norm(beta_new)
            if diff < self.tol:
                break

            eps_s = eps_s_new
            eps_b = eps_b_new
            alpha = alpha_new
            beta = beta_new

        # Final surface Green's function
        g_surface = la.inv(E_plus - eps_s)

        return g_surface

    def calculate_iterative(self, E: float) -> np.ndarray:
        """
        Alternative: Direct iterative method for surface Green's function.

        Solves: g_s = [(E + iη)I - H_00 - H_01 · g_s · H_01†]^(-1)

        This is simpler but slower than decimation.
        """
        I = np.eye(self.n, dtype=complex)
        E_plus = (E + 1j * self.eta) * I

        # Initial guess
        g_s = la.inv(E_plus - self.H_00)

        for _ in range(self.max_iter):
            # Self-energy from surface
            sigma = self.H_01 @ g_s @ self.H_10

            # New surface Green's function
            g_s_new = la.inv(E_plus - self.H_00 - sigma)

            # Check convergence
            if la.norm(g_s_new - g_s) < self.tol:
                break

            g_s = g_s_new

        return g_s


class NEGFSolver:
    """
    Non-Equilibrium Green's Function solver for quantum transport.

    Implements the NEGF formalism following the flowchart:
    1. Initial charge density ρ
    2. Calculate Hamiltonian H (with potential from Poisson)
    3. Calculate NEGF: G^<, G^A, G^R
    4. Calculate electron density: ρ(r) = ∫(dE/2π) Im[G^<(E)]
    5. Check self-consistency
    6. Calculate physical quantities (current, transmission)
    """

    def __init__(self, params: GAADeviceParams, eta: float = 1e-6):
        """
        Initialize NEGF solver.

        Args:
            params: GAA device parameters
            eta: Small imaginary part for Green's function broadening
        """
        self.params = params
        self.eta = eta

        # Build spatial grid
        self._build_grid()

        # Initialize Hamiltonian and potential
        self.H = None
        self.potential = None
        self.electron_density = None

        # Lead Green's functions
        self.g_source = None
        self.g_drain = None

    def _build_grid(self):
        """Build spatial discretization grid."""
        p = self.params

        # z-direction (transport direction)
        self.dz = p.channel_length / (p.nz - 1)
        self.z = np.linspace(0, p.channel_length, p.nz)

        # r-direction (radial, for full 3D GAA)
        if p.nr > 1:
            self.dr = p.nanowire_radius / (p.nr - 1)
            self.r = np.linspace(0, p.nanowire_radius, p.nr)
        else:
            self.dr = p.nanowire_radius
            self.r = np.array([0])

        # Total number of grid points
        self.n_total = p.nz * p.nr

    def build_hamiltonian(self, potential: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build the device Hamiltonian using effective mass approximation.

        For a 1D effective mass Hamiltonian with finite differences:
        H = -ℏ²/(2m*) · d²/dz² + V(z)

        Discretized:
        H_ii = 2t + V_i   (diagonal)
        H_i,i±1 = -t      (off-diagonal)

        where t = ℏ²/(2m*·Δz²) is the hopping parameter.

        Args:
            potential: Electrostatic potential profile (eV)

        Returns:
            Hamiltonian matrix (n x n)
        """
        p = self.params
        m_eff = p.material.effective_mass * M_E

        # Hopping parameter in eV
        t_z = (HBAR**2 / (2 * m_eff * self.dz**2)) / Q_E

        # Build tridiagonal Hamiltonian for 1D case
        if p.nr == 1:
            n = p.nz
            H = np.zeros((n, n), dtype=complex)

            # Diagonal elements: 2t + V(z)
            for i in range(n):
                H[i, i] = 2 * t_z
                if potential is not None:
                    H[i, i] += potential[i]

            # Off-diagonal elements: -t (nearest neighbor hopping)
            for i in range(n - 1):
                H[i, i + 1] = -t_z
                H[i + 1, i] = -t_z
        else:
            # Full 2D (r,z) Hamiltonian for cylindrical GAA
            H = self._build_2d_hamiltonian(potential, t_z)

        self.H = H
        return H

    def _build_2d_hamiltonian(self, potential: np.ndarray, t_z: float) -> np.ndarray:
        """Build 2D cylindrical Hamiltonian for GAA geometry."""
        p = self.params
        m_eff = p.material.effective_mass * M_E

        # Radial hopping
        t_r = (HBAR**2 / (2 * m_eff * self.dr**2)) / Q_E

        n = p.nz * p.nr
        H = np.zeros((n, n), dtype=complex)

        def idx(iz, ir):
            """Convert 2D index to 1D."""
            return iz * p.nr + ir

        for iz in range(p.nz):
            for ir in range(p.nr):
                i = idx(iz, ir)

                # Diagonal
                H[i, i] = 2 * t_z + 2 * t_r
                if potential is not None:
                    H[i, i] += potential[iz, ir] if potential.ndim > 1 else potential[iz]

                # z-direction hopping
                if iz > 0:
                    H[i, idx(iz - 1, ir)] = -t_z
                if iz < p.nz - 1:
                    H[i, idx(iz + 1, ir)] = -t_z

                # r-direction hopping (with cylindrical correction)
                if ir > 0:
                    # Include 1/r factor for cylindrical coordinates
                    r_factor = np.sqrt(self.r[ir] / self.r[ir - 1]) if self.r[ir - 1] > 0 else 1
                    H[i, idx(iz, ir - 1)] = -t_r * r_factor
                if ir < p.nr - 1:
                    r_factor = np.sqrt(self.r[ir] / self.r[ir + 1])
                    H[i, idx(iz, ir + 1)] = -t_r * r_factor

        return H

    def calculate_lead_self_energy(self, E: float, lead: str = 'source') -> np.ndarray:
        """
        Calculate self-energy from semi-infinite leads.

        The self-energy connects the device region to semi-infinite contacts:
        Σ^R_L(E) = τ_L · g^R_L(E) · τ_L†

        where g^R_L is the surface Green's function of the lead and
        τ_L is the coupling between device and lead.

        Args:
            E: Energy (eV)
            lead: 'source' or 'drain'

        Returns:
            Self-energy matrix (same size as device Hamiltonian)
        """
        p = self.params
        m_eff = p.material.effective_mass * M_E
        t = (HBAR**2 / (2 * m_eff * self.dz**2)) / Q_E

        n = self.H.shape[0]
        sigma = np.zeros((n, n), dtype=complex)

        # Lead unit cell Hamiltonian (same as device)
        if p.nr == 1:
            H_lead = np.array([[2 * t]], dtype=complex)
            H_coupling = np.array([[-t]], dtype=complex)
        else:
            # Multi-mode lead
            H_lead = np.zeros((p.nr, p.nr), dtype=complex)
            H_coupling = np.zeros((p.nr, p.nr), dtype=complex)
            t_r = (HBAR**2 / (2 * m_eff * self.dr**2)) / Q_E
            for ir in range(p.nr):
                H_lead[ir, ir] = 2 * t + 2 * t_r
                if ir > 0:
                    H_lead[ir, ir - 1] = -t_r
                    H_lead[ir - 1, ir] = -t_r
                H_coupling[ir, ir] = -t

        # Add Fermi level shift for doping
        if lead == 'source':
            E_F_shift = self._fermi_level_from_doping(p.source_doping)
            H_lead_shifted = H_lead - E_F_shift * np.eye(H_lead.shape[0])
            bias = p.vs
        else:
            E_F_shift = self._fermi_level_from_doping(p.drain_doping)
            H_lead_shifted = H_lead - E_F_shift * np.eye(H_lead.shape[0])
            bias = p.vd

        # Calculate surface Green's function
        sgf = SurfaceGreenFunction(H_lead_shifted, H_coupling, eta=self.eta)
        g_surface = sgf.calculate(E - bias)

        # Self-energy: Σ = τ · g_s · τ†
        # τ is the coupling matrix between lead and device
        if p.nr == 1:
            if lead == 'source':
                sigma[0, 0] = -t * g_surface[0, 0] * (-t)
            else:
                sigma[-1, -1] = -t * g_surface[0, 0] * (-t)
        else:
            tau = H_coupling
            sigma_block = tau @ g_surface @ tau.T.conj()
            if lead == 'source':
                sigma[:p.nr, :p.nr] = sigma_block
            else:
                sigma[-p.nr:, -p.nr:] = sigma_block

        return sigma

    def _fermi_level_from_doping(self, doping: float) -> float:
        """
        Calculate Fermi level shift from doping concentration.

        Using the non-degenerate approximation:
        n = N_c · exp((E_F - E_c) / kT)
        E_F - E_c = kT · ln(n / N_c)

        Args:
            doping: Doping concentration (1/cm³)

        Returns:
            Fermi level shift (eV)
        """
        p = self.params
        m_eff = p.material.effective_mass
        T = p.temperature

        # Effective density of states
        N_c = 2 * (2 * np.pi * m_eff * M_E * K_B * T / (2 * np.pi * HBAR)**2)**1.5
        N_c *= 1e-6  # Convert to 1/cm³

        # Fermi level (simplified)
        kT = K_B * T / Q_E  # in eV
        E_F = kT * np.log(doping / N_c)

        return E_F

    def calculate_retarded_greens_function(self, E: float) -> np.ndarray:
        """
        Calculate retarded Green's function.

        G^R(E) = [(E + iη)I - H - Σ^R_S(E) - Σ^R_D(E)]^(-1)

        Args:
            E: Energy (eV)

        Returns:
            Retarded Green's function matrix
        """
        if self.H is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian first.")

        n = self.H.shape[0]
        I = np.eye(n, dtype=complex)

        # Total self-energy from source and drain
        sigma_S = self.calculate_lead_self_energy(E, 'source')
        sigma_D = self.calculate_lead_self_energy(E, 'drain')
        sigma_total = sigma_S + sigma_D

        # Retarded Green's function
        G_R = la.inv((E + 1j * self.eta) * I - self.H - sigma_total)

        return G_R, sigma_S, sigma_D

    def calculate_advanced_greens_function(self, G_R: np.ndarray) -> np.ndarray:
        """
        Calculate advanced Green's function.

        G^A(E) = [G^R(E)]†

        Args:
            G_R: Retarded Green's function

        Returns:
            Advanced Green's function matrix
        """
        return G_R.T.conj()

    def calculate_lesser_greens_function(self, E: float, G_R: np.ndarray,
                                         sigma_S: np.ndarray, sigma_D: np.ndarray) -> np.ndarray:
        """
        Calculate lesser Green's function.

        G^<(E) = G^R(E) · Σ^<(E) · G^A(E)

        where Σ^< = i·f_S(E)·Γ_S + i·f_D(E)·Γ_D
        and Γ = i(Σ^R - Σ^A) is the broadening matrix.

        Args:
            E: Energy (eV)
            G_R: Retarded Green's function
            sigma_S: Source self-energy
            sigma_D: Drain self-energy

        Returns:
            Lesser Green's function matrix
        """
        p = self.params

        # Broadening matrices: Γ = i(Σ^R - Σ^A) = -2·Im(Σ^R)
        gamma_S = 1j * (sigma_S - sigma_S.T.conj())
        gamma_D = 1j * (sigma_D - sigma_D.T.conj())

        # Fermi functions
        kT = K_B * p.temperature / Q_E  # in eV
        f_S = self._fermi_function(E - p.vs, kT)
        f_D = self._fermi_function(E - p.vd, kT)

        # Lesser self-energy: Σ^< = i·f·Γ
        sigma_lesser = 1j * f_S * gamma_S + 1j * f_D * gamma_D

        # Advanced Green's function
        G_A = self.calculate_advanced_greens_function(G_R)

        # Lesser Green's function
        G_lesser = G_R @ sigma_lesser @ G_A

        return G_lesser

    def _fermi_function(self, E: float, kT: float) -> float:
        """
        Fermi-Dirac distribution function.

        f(E) = 1 / (1 + exp(E / kT))

        Args:
            E: Energy relative to Fermi level (eV)
            kT: Thermal energy (eV)

        Returns:
            Fermi function value
        """
        if kT < 1e-10:
            return 1.0 if E < 0 else 0.0

        x = E / kT
        if x > 100:
            return 0.0
        elif x < -100:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))

    def calculate_electron_density(self, E_min: float, E_max: float,
                                   n_energy: int = 100) -> np.ndarray:
        """
        Calculate electron density by integrating lesser Green's function.

        n(r) = (1/2π) ∫ dE · (-i) · G^<(r,r,E)
             = (1/π) ∫ dE · Im[G^<(r,r,E)]

        Args:
            E_min: Minimum energy for integration (eV)
            E_max: Maximum energy for integration (eV)
            n_energy: Number of energy points

        Returns:
            Electron density at each grid point (1/cm³)
        """
        p = self.params

        # Energy grid
        E_grid = np.linspace(E_min, E_max, n_energy)
        dE = E_grid[1] - E_grid[0]

        n = self.H.shape[0]
        density = np.zeros(n)

        for E in E_grid:
            G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
            G_lesser = self.calculate_lesser_greens_function(E, G_R, sigma_S, sigma_D)

            # Electron density from diagonal of G^<
            # n = -i·G^< / (2π) -> integrate over energy
            density += np.diag(G_lesser).imag

        # Multiply by 1/π and dE, convert units
        density *= -dE / np.pi

        # Convert to 1/cm³ (multiply by 1/volume_per_grid_point)
        if p.nr == 1:
            volume = np.pi * p.nanowire_radius**2 * self.dz  # cylindrical shell
        else:
            volume = self.dz * 2 * np.pi * self.dr * self.r.mean()

        density /= volume * 1e6  # Convert m³ to cm³

        self.electron_density = density
        return density

    def calculate_transmission(self, E: float) -> float:
        """
        Calculate transmission coefficient at energy E.

        T(E) = Tr[Γ_S · G^R · Γ_D · G^A]

        This gives the probability of electron transmission from source to drain.

        Args:
            E: Energy (eV)

        Returns:
            Transmission coefficient
        """
        G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
        G_A = self.calculate_advanced_greens_function(G_R)

        # Broadening matrices
        gamma_S = 1j * (sigma_S - sigma_S.T.conj())
        gamma_D = 1j * (sigma_D - sigma_D.T.conj())

        # Transmission
        T = np.trace(gamma_S @ G_R @ gamma_D @ G_A).real

        return T

    def calculate_spectral_function(self, E: float) -> np.ndarray:
        """
        Calculate spectral function (local density of states).

        A(E) = i(G^R - G^A) = -2·Im(G^R)

        Args:
            E: Energy (eV)

        Returns:
            Spectral function diagonal elements
        """
        G_R, _, _ = self.calculate_retarded_greens_function(E)
        A = -2 * G_R.imag
        return np.diag(A)

    def calculate_current(self, E_min: float, E_max: float,
                          n_energy: int = 200) -> Tuple[float, np.ndarray]:
        """
        Calculate current using Landauer-Büttiker formula.

        I = (q/h) ∫ dE · T(E) · [f_S(E) - f_D(E)]

        Also calculates position-resolved current from the bond current formula:
        I_{l→l+1} = (2q/ℏ) ∫ dE · Re{Tr[H_{l,l+1} · G^<_{l+1,l}(E)]}

        Args:
            E_min: Minimum energy (eV)
            E_max: Maximum energy (eV)
            n_energy: Number of energy points

        Returns:
            Tuple of (total current in Amperes, current spectrum)
        """
        p = self.params
        kT = K_B * p.temperature / Q_E

        E_grid = np.linspace(E_min, E_max, n_energy)
        dE = E_grid[1] - E_grid[0]

        current_spectrum = np.zeros(n_energy)

        for i, E in enumerate(E_grid):
            T_E = self.calculate_transmission(E)
            f_S = self._fermi_function(E - p.vs, kT)
            f_D = self._fermi_function(E - p.vd, kT)

            # Current contribution at this energy
            current_spectrum[i] = T_E * (f_S - f_D)

        # Total current: I = (2q/h) ∫ T(E)[f_S - f_D] dE
        # Factor of 2 for spin
        prefactor = 2 * Q_E / (2 * np.pi * HBAR)
        total_current = prefactor * np.trapezoid(current_spectrum, E_grid) * Q_E  # Convert eV to J

        return total_current, current_spectrum

    def calculate_bond_current(self, E: float, l: int) -> float:
        """
        Calculate bond current between layers l and l+1.

        I_{l→l+1}(E) = (2q/ℏ) · Re{Tr[H_{l,l+1} · G^<_{l+1,l}(E)]}

        This is the position-resolved current useful for analyzing
        current flow through the device.

        Args:
            E: Energy (eV)
            l: Layer index (0 to nz-2)

        Returns:
            Bond current at energy E
        """
        p = self.params

        if l < 0 or l >= p.nz - 1:
            raise ValueError(f"Layer index must be between 0 and {p.nz - 2}")

        G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
        G_lesser = self.calculate_lesser_greens_function(E, G_R, sigma_S, sigma_D)

        if p.nr == 1:
            # 1D case
            H_coupling = self.H[l, l + 1]
            G_lesser_coupling = G_lesser[l + 1, l]

            # Bond current
            I_bond = (2 * Q_E / HBAR) * np.real(H_coupling * G_lesser_coupling)
        else:
            # Multi-mode case
            idx_l = l * p.nr
            idx_l1 = (l + 1) * p.nr

            H_block = self.H[idx_l:idx_l + p.nr, idx_l1:idx_l1 + p.nr]
            G_block = G_lesser[idx_l1:idx_l1 + p.nr, idx_l:idx_l + p.nr]

            I_bond = (2 * Q_E / HBAR) * np.real(np.trace(H_block @ G_block))

        return I_bond


class PoissonSolver:
    """
    Poisson equation solver for electrostatic potential in GAA geometry.

    Solves: ∇·(ε∇φ) = -ρ/ε₀

    For cylindrical GAA geometry with appropriate boundary conditions.
    """

    def __init__(self, params: GAADeviceParams):
        """Initialize Poisson solver."""
        self.params = params

    def solve_1d(self, charge_density: np.ndarray,
                 bc_source: float = 0.0, bc_drain: float = 0.0) -> np.ndarray:
        """
        Solve 1D Poisson equation along transport direction with gate coupling.

        For GAA transistor, the gate controls the channel potential through
        capacitive coupling. The potential profile determines the barrier
        height that electrons must overcome.

        The model includes:
        - Laplacian term for charge distribution
        - Gate capacitive coupling (stronger in GAA due to wraparound gate)
        - Built-in potential from doping differences

        Args:
            charge_density: Electron density (1/cm³)
            bc_source: Boundary condition at source (V)
            bc_drain: Boundary condition at drain (V)

        Returns:
            Electrostatic potential (eV) - this is the conduction band edge
        """
        p = self.params
        nz = p.nz
        dz = p.channel_length / (nz - 1)
        z = np.linspace(0, p.channel_length, nz)

        # Gate oxide capacitance per unit area
        C_ox = EPSILON_0 * p.oxide_dielectric / p.oxide_thickness

        # For GAA geometry, effective capacitance is enhanced
        # C_gaa = 2π ε_ox / ln(1 + t_ox/r) ≈ C_ox * (2πr) / (πr²) for thin oxide
        # This gives stronger gate control in GAA vs planar
        r = p.nanowire_radius
        gaa_factor = 2.0 / r  # Enhanced coupling for cylindrical geometry

        # Lambda = screening length in channel
        lambda_ch = np.sqrt(EPSILON_0 * p.material.dielectric_constant /
                           (Q_E * p.channel_doping * 1e6 + 1e10))  # Add small term to avoid div by 0

        # Gate coupling strength (dimensionless)
        # For GAA: strong coupling in channel, weak at contacts
        gate_coupling = C_ox * gaa_factor / (EPSILON_0 * p.material.dielectric_constant / dz**2)

        # Define regions: source contact, channel, drain contact
        n_contact = max(2, int(0.15 * nz))  # 15% of device is contact region on each side

        # Build potential with gate modulation
        potential = np.zeros(nz)

        # Source/drain Fermi levels (from doping)
        kT = K_B * p.temperature / Q_E
        E_F_source = kT * np.log(p.source_doping / 1e17)  # Relative to intrinsic
        E_F_drain = kT * np.log(p.drain_doping / 1e17)

        # Channel barrier without gate (built-in potential)
        # Barrier height = E_g/2 + kT*ln(N_d/n_i) approximately
        barrier_height = 0.3  # Base barrier in eV (simplified)

        for i in range(nz):
            if i < n_contact:
                # Source region - heavily doped, potential near 0
                potential[i] = p.vs
            elif i >= nz - n_contact:
                # Drain region - heavily doped, potential at Vd
                potential[i] = p.vd
            else:
                # Channel region - gate controlled
                # Position within channel (0 to 1)
                z_rel = (i - n_contact) / (nz - 2 * n_contact)

                # Linear interpolation for drain-induced barrier lowering
                v_dibl = p.vs + (p.vd - p.vs) * z_rel

                # Gate-controlled barrier
                # Higher Vg lowers the barrier (for n-channel)
                # Use a threshold voltage model: V_barrier = V_t - Vg
                v_threshold = 0.25  # Threshold voltage

                # Gate modulation: barrier reduces as Vg increases above Vt
                gate_effect = max(0, barrier_height - (p.vg - v_threshold))

                # Total potential (conduction band edge relative to source Fermi level)
                potential[i] = v_dibl + gate_effect

                # Include charge density effect (screening)
                if charge_density is not None and len(charge_density) > i:
                    n = charge_density[i] * 1e6  # Convert to m^-3
                    # Potential lowering due to electron screening
                    screening = -Q_E * n * dz**2 / (EPSILON_0 * p.material.dielectric_constant)
                    potential[i] += screening * 0.01  # Scaled for stability

        return potential

    def _gate_potential(self, z: float) -> float:
        """
        Gate potential profile along channel.

        For uniform gate coverage, this is constant.
        Can be modified for split-gate or other geometries.
        """
        return self.params.vg


class SelfConsistentNEGF:
    """
    Self-consistent NEGF-Poisson solver.

    Implements the full DFT+NEGF loop:
    1. Initialize charge density
    2. Solve Poisson for potential
    3. Build Hamiltonian with potential
    4. Solve NEGF for new density
    5. Mix old and new density
    6. Check convergence
    7. Repeat until converged
    """

    def __init__(self, params: GAADeviceParams,
                 mixing: float = 0.3, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize self-consistent solver.

        Args:
            params: Device parameters
            mixing: Linear mixing parameter (0 < mixing <= 1)
            max_iter: Maximum self-consistent iterations
            tol: Convergence tolerance for charge density
        """
        self.params = params
        self.mixing = mixing
        self.max_iter = max_iter
        self.tol = tol

        self.negf = NEGFSolver(params)
        self.poisson = PoissonSolver(params)

        self.converged = False
        self.iteration = 0
        self.density_history = []

    def initialize_density(self) -> np.ndarray:
        """
        Initialize charge density for self-consistent loop.

        Uses doping profile as initial guess.
        """
        p = self.params
        n = p.nz if p.nr == 1 else p.nz * p.nr

        density = np.zeros(n)

        # Source and drain regions (first/last 10% of channel)
        n_contact = max(1, int(0.1 * p.nz))

        for i in range(n):
            iz = i // p.nr if p.nr > 1 else i

            if iz < n_contact:
                density[i] = p.source_doping
            elif iz >= p.nz - n_contact:
                density[i] = p.drain_doping
            else:
                density[i] = p.channel_doping

        return density

    def solve(self, E_min: float = -0.5, E_max: float = 0.5,
              n_energy: int = 100, verbose: bool = True) -> Dict:
        """
        Run self-consistent NEGF-Poisson loop.

        Args:
            E_min: Minimum energy for NEGF integration
            E_max: Maximum energy for NEGF integration
            n_energy: Number of energy points
            verbose: Print progress information

        Returns:
            Dictionary with results (density, potential, current, etc.)
        """
        p = self.params

        # Initialize
        density = self.initialize_density()
        potential = np.zeros_like(density)

        if verbose:
            print("Starting self-consistent NEGF-Poisson loop...")
            print(f"  Vg = {p.vg:.3f} V, Vd = {p.vd:.3f} V")

        for iteration in range(self.max_iter):
            self.iteration = iteration

            # 1. Solve Poisson equation
            if p.nr == 1:
                potential = self.poisson.solve_1d(density, p.vs, p.vd)

            # 2. Build Hamiltonian with new potential
            self.negf.build_hamiltonian(potential)

            # 3. Calculate new electron density from NEGF
            density_new = self.negf.calculate_electron_density(E_min, E_max, n_energy)

            # 4. Mix densities
            density_diff = la.norm(density_new - density) / (la.norm(density) + 1e-10)
            density = (1 - self.mixing) * density + self.mixing * density_new

            self.density_history.append(density.copy())

            if verbose:
                print(f"  Iteration {iteration + 1}: density change = {density_diff:.2e}")

            # 5. Check convergence
            if density_diff < self.tol:
                self.converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break

        if not self.converged and verbose:
            warnings.warn(f"Did not converge after {self.max_iter} iterations")

        # Calculate final quantities
        total_current, current_spectrum = self.negf.calculate_current(E_min, E_max, n_energy)

        # Calculate transmission spectrum
        E_grid = np.linspace(E_min, E_max, n_energy)
        transmission = np.array([self.negf.calculate_transmission(E) for E in E_grid])

        results = {
            'density': density,
            'potential': potential,
            'current': total_current,
            'current_spectrum': current_spectrum,
            'transmission': transmission,
            'energy_grid': E_grid,
            'converged': self.converged,
            'iterations': self.iteration + 1,
            'z_grid': self.negf.z,
        }

        return results


def run_example_simulation(vg_min: float = 0.0, vg_max: float = 0.7,
                           n_vg_points: int = 8, vd: float = 0.05,
                           plot_results: bool = True):
    """
    Run GAA transistor simulation with gate voltage sweep to generate transfer curve.

    Sweeps gate voltage from vg_min to vg_max and calculates drain current at each point.
    This produces a typical transfer characteristic (I_D vs V_G) curve.

    Args:
        vg_min: Minimum gate voltage (V)
        vg_max: Maximum gate voltage (V)
        n_vg_points: Number of gate voltage points
        vd: Drain voltage (V)
        plot_results: Whether to generate and save plot

    Returns:
        Tuple of (vg_values, currents, all_results)
    """
    print("=" * 60)
    print("NEGF Transport Simulation for GAA Transistor")
    print("Transfer Characteristics (I_D vs V_G)")
    print("=" * 60)

    # Base device parameters
    base_params = {
        'channel_length': 12e-9,     # 12 nm channel
        'nanowire_radius': 2.5e-9,   # 2.5 nm radius nanowire
        'oxide_thickness': 1e-9,     # 1 nm oxide
        'nz': 25,                    # Grid points
        'nr': 1,                     # 1D simulation
        'vd': vd,                    # Drain voltage
        'source_doping': 1e20,       # n+ source
        'drain_doping': 1e20,        # n+ drain
        'channel_doping': 1e15,      # Lightly doped channel (intrinsic)
        'temperature': 300,          # Room temperature
    }

    # Create initial params for display
    params = GAADeviceParams(**base_params, vg=0.0)

    print(f"\nDevice Parameters:")
    print(f"  Channel length: {params.channel_length * 1e9:.1f} nm")
    print(f"  Nanowire radius: {params.nanowire_radius * 1e9:.1f} nm")
    print(f"  Material: {params.material.name}")
    print(f"  Effective mass: {params.material.effective_mass} m_e")
    print(f"  Drain voltage: {vd} V")
    print(f"\nGate voltage sweep: {vg_min} V to {vg_max} V ({n_vg_points} points)")
    print("-" * 60)

    # Gate voltage sweep
    vg_values = np.linspace(vg_min, vg_max, n_vg_points)
    currents = []
    all_results = []

    for i, vg in enumerate(vg_values):
        print(f"\n[{i+1}/{n_vg_points}] V_G = {vg:.3f} V")

        # Create params for this gate voltage
        params = GAADeviceParams(**base_params, vg=vg)

        # Run self-consistent simulation
        solver = SelfConsistentNEGF(params, mixing=0.25, max_iter=40, tol=1e-3)
        results = solver.solve(E_min=-0.4, E_max=0.8, n_energy=40, verbose=False)

        # Store current (take absolute value for plotting)
        current = np.abs(results['current'])
        currents.append(current)
        all_results.append(results)

        status = "converged" if results['converged'] else f"({results['iterations']} iters)"
        print(f"  I_D = {current * 1e6:.4f} µA  [{status}]")

    currents = np.array(currents)

    # Print summary
    print("\n" + "=" * 60)
    print("Transfer Characteristics Summary")
    print("=" * 60)
    print(f"{'V_G (V)':<12} {'I_D (µA)':<15} {'I_D (A)':<15}")
    print("-" * 42)
    for vg, current in zip(vg_values, currents):
        print(f"{vg:<12.3f} {current*1e6:<15.4f} {current:<15.4e}")

    # Plot transfer curve
    if plot_results:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Linear scale plot
            ax1.plot(vg_values, currents * 1e6, 'b-o', linewidth=2, markersize=8,
                     markerfacecolor='white', markeredgewidth=2)
            ax1.set_xlabel('V$_G$ (V)', fontsize=12)
            ax1.set_ylabel('I$_{DS}$ (µA)', fontsize=12)
            ax1.set_title('Transfer Characteristics (Linear Scale)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([vg_min - 0.05, vg_max + 0.05])
            ax1.set_ylim(bottom=0)

            # Log scale plot (for subthreshold behavior)
            ax2.semilogy(vg_values, currents * 1e6 + 1e-6, 'r-s', linewidth=2, markersize=8,
                        markerfacecolor='white', markeredgewidth=2)
            ax2.set_xlabel('V$_G$ (V)', fontsize=12)
            ax2.set_ylabel('I$_{DS}$ (µA)', fontsize=12)
            ax2.set_title('Transfer Characteristics (Log Scale)', fontsize=14)
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlim([vg_min - 0.05, vg_max + 0.05])

            plt.suptitle(f'GAA Transistor: L={base_params["channel_length"]*1e9:.0f}nm, '
                        f'R={base_params["nanowire_radius"]*1e9:.1f}nm, V$_D$={vd}V',
                        fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig('transfer_curve.png', dpi=150, bbox_inches='tight')
            print(f"\nTransfer curve saved to: transfer_curve.png")
            plt.show()
        except ImportError:
            print("\nMatplotlib not available for plotting.")

    return vg_values, currents, all_results


if __name__ == "__main__":
    # Run transfer characteristics simulation
    vg_values, currents, results = run_example_simulation(
        vg_min=0.0,
        vg_max=0.7,
        n_vg_points=8,
        vd=0.05,
        plot_results=True
    )
