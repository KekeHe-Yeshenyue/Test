#!/usr/bin/env python3
"""
Example script for running NEGF simulations on a GAA transistor.

This script demonstrates:
1. Single bias point simulation
2. Transfer characteristics (I_D vs V_G)
3. Output characteristics (I_D vs V_D)
4. Visualization of internal quantities

The NEGF formalism calculates quantum transport including:
- Tunneling through barriers
- Quantum confinement effects
- Non-equilibrium carrier distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from negf_transport import (
    GAADeviceParams, NEGFSolver, SelfConsistentNEGF,
    Material, MaterialType
)


def single_point_analysis():
    """
    Detailed analysis at a single bias point.

    Shows the internal NEGF quantities:
    - Transmission spectrum T(E)
    - Local density of states (LDOS)
    - Potential profile
    - Electron density distribution
    """
    print("\n" + "=" * 60)
    print("Single Bias Point Analysis")
    print("=" * 60)

    # Device setup
    params = GAADeviceParams(
        channel_length=12e-9,
        nanowire_radius=2.5e-9,
        nz=40,
        nr=1,
        vg=0.4,
        vd=0.1,
        source_doping=1e20,
        drain_doping=1e20,
        channel_doping=1e16,
        temperature=300,
    )

    # Run self-consistent simulation
    solver = SelfConsistentNEGF(params, mixing=0.15, max_iter=30, tol=1e-3)
    results = solver.solve(E_min=-0.5, E_max=0.8, n_energy=80, verbose=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Transmission spectrum
    ax1 = axes[0, 0]
    E_grid = results['energy_grid']
    ax1.plot(E_grid, results['transmission'], 'b-', linewidth=2)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission T(E)')
    ax1.set_title('Transmission Spectrum')
    ax1.set_ylim([0, max(1.2, results['transmission'].max() * 1.1)])
    ax1.grid(True, alpha=0.3)

    # 2. Potential profile
    ax2 = axes[0, 1]
    z_nm = results['z_grid'] * 1e9
    ax2.plot(z_nm, results['potential'], 'r-', linewidth=2)
    ax2.set_xlabel('Position z (nm)')
    ax2.set_ylabel('Potential (V)')
    ax2.set_title('Electrostatic Potential Profile')
    ax2.grid(True, alpha=0.3)

    # 3. Electron density
    ax3 = axes[1, 0]
    ax3.semilogy(z_nm, np.abs(results['density']) + 1e10, 'g-', linewidth=2)
    ax3.set_xlabel('Position z (nm)')
    ax3.set_ylabel('Electron Density (cm⁻³)')
    ax3.set_title('Electron Density Distribution')
    ax3.grid(True, alpha=0.3)

    # 4. Current spectrum
    ax4 = axes[1, 1]
    ax4.plot(E_grid, results['current_spectrum'], 'm-', linewidth=2)
    ax4.fill_between(E_grid, results['current_spectrum'], alpha=0.3)
    ax4.set_xlabel('Energy (eV)')
    ax4.set_ylabel('Current Spectrum (arb. units)')
    ax4.set_title('Energy-resolved Current')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'GAA Transistor: Vg={params.vg}V, Vd={params.vd}V\n'
                 f'Current = {results["current"]*1e6:.4f} µA', fontsize=12)
    plt.tight_layout()
    plt.savefig('single_point_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def calculate_transfer_characteristics():
    """
    Calculate transfer characteristics: I_D vs V_G at fixed V_D.

    This shows how the transistor current varies with gate voltage,
    demonstrating ON/OFF switching behavior.
    """
    print("\n" + "=" * 60)
    print("Transfer Characteristics (I_D vs V_G)")
    print("=" * 60)

    # Base device parameters
    base_params = {
        'channel_length': 10e-9,
        'nanowire_radius': 2.5e-9,
        'nz': 30,
        'nr': 1,
        'vd': 0.05,  # Low drain bias for linear region
        'source_doping': 1e20,
        'drain_doping': 1e20,
        'channel_doping': 1e16,
        'temperature': 300,
    }

    # Gate voltage sweep
    vg_values = np.linspace(-0.2, 0.6, 9)
    currents = []

    for i, vg in enumerate(vg_values):
        print(f"\nVg = {vg:.2f} V ({i+1}/{len(vg_values)})")

        params = GAADeviceParams(**base_params, vg=vg)
        solver = SelfConsistentNEGF(params, mixing=0.2, max_iter=20, tol=1e-3)
        results = solver.solve(E_min=-0.4, E_max=0.6, n_energy=40, verbose=False)

        currents.append(results['current'])
        print(f"  Current: {results['current']*1e6:.4f} µA")

    currents = np.array(currents)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.plot(vg_values, currents * 1e6, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Gate Voltage V_G (V)')
    ax1.set_ylabel('Drain Current I_D (µA)')
    ax1.set_title('Transfer Characteristics (Linear Scale)')
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(vg_values, np.abs(currents) * 1e6 + 1e-6, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Gate Voltage V_G (V)')
    ax2.set_ylabel('Drain Current I_D (µA)')
    ax2.set_title('Transfer Characteristics (Log Scale)')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'GAA Transistor Transfer Characteristics\nV_D = {base_params["vd"]} V')
    plt.tight_layout()
    plt.savefig('transfer_characteristics.png', dpi=150, bbox_inches='tight')
    plt.show()

    return vg_values, currents


def calculate_output_characteristics():
    """
    Calculate output characteristics: I_D vs V_D at different V_G values.

    This shows the current-voltage relationship demonstrating
    linear and saturation regions.
    """
    print("\n" + "=" * 60)
    print("Output Characteristics (I_D vs V_D)")
    print("=" * 60)

    # Base device parameters
    base_params = {
        'channel_length': 10e-9,
        'nanowire_radius': 2.5e-9,
        'nz': 25,
        'nr': 1,
        'source_doping': 1e20,
        'drain_doping': 1e20,
        'channel_doping': 1e16,
        'temperature': 300,
    }

    # Gate voltages to simulate
    vg_list = [0.2, 0.3, 0.4, 0.5]
    vd_values = np.linspace(0.01, 0.3, 7)

    results_dict = {}

    for vg in vg_list:
        print(f"\n--- V_G = {vg} V ---")
        currents = []

        for vd in vd_values:
            print(f"  V_D = {vd:.2f} V...", end=" ")

            params = GAADeviceParams(**base_params, vg=vg, vd=vd)
            solver = SelfConsistentNEGF(params, mixing=0.2, max_iter=15, tol=1e-3)
            results = solver.solve(E_min=-0.4, E_max=0.6, n_energy=30, verbose=False)

            currents.append(results['current'])
            print(f"I = {results['current']*1e6:.4f} µA")

        results_dict[vg] = np.array(currents)

    # Plot
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vg_list)))

    for vg, color in zip(vg_list, colors):
        plt.plot(vd_values, results_dict[vg] * 1e6, 'o-',
                 color=color, linewidth=2, markersize=8, label=f'V_G = {vg} V')

    plt.xlabel('Drain Voltage V_D (V)', fontsize=12)
    plt.ylabel('Drain Current I_D (µA)', fontsize=12)
    plt.title('GAA Transistor Output Characteristics', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output_characteristics.png', dpi=150, bbox_inches='tight')
    plt.show()

    return vd_values, results_dict


def demonstrate_ldos():
    """
    Calculate and visualize the Local Density of States (LDOS).

    LDOS(z, E) = -(1/π) Im[G^R(z,z,E)]

    This shows how quantum states are distributed in energy and space.
    """
    print("\n" + "=" * 60)
    print("Local Density of States (LDOS) Analysis")
    print("=" * 60)

    params = GAADeviceParams(
        channel_length=15e-9,
        nanowire_radius=2.5e-9,
        nz=50,
        nr=1,
        vg=0.3,
        vd=0.2,
        source_doping=1e20,
        drain_doping=1e20,
        channel_doping=1e16,
        temperature=300,
    )

    # Build device and run initial self-consistency
    solver = SelfConsistentNEGF(params, mixing=0.2, max_iter=20, tol=1e-3)
    results = solver.solve(E_min=-0.5, E_max=0.8, n_energy=50, verbose=True)

    # Calculate LDOS on fine energy grid
    negf = solver.negf
    E_fine = np.linspace(-0.4, 0.7, 80)
    z_nm = negf.z * 1e9

    LDOS = np.zeros((len(E_fine), params.nz))

    print("Calculating LDOS...")
    for i, E in enumerate(E_fine):
        A = negf.calculate_spectral_function(E)
        LDOS[i, :] = A / np.pi

    # Create 2D plot
    plt.figure(figsize=(12, 8))

    # Use log scale for better visualization
    LDOS_plot = np.log10(np.abs(LDOS) + 1e-10)

    plt.pcolormesh(z_nm, E_fine, LDOS_plot, shading='auto', cmap='hot')
    plt.colorbar(label='log₁₀(LDOS)')

    # Add Fermi levels
    plt.axhline(y=params.vs, color='cyan', linestyle='--', linewidth=2,
                label=f'E_F (source) = {params.vs} eV')
    plt.axhline(y=params.vd, color='lime', linestyle='--', linewidth=2,
                label=f'E_F (drain) = {params.vd} eV')

    plt.xlabel('Position z (nm)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'Local Density of States\nV_G = {params.vg} V, V_D = {params.vd} V', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig('ldos_map.png', dpi=150, bbox_inches='tight')
    plt.show()

    return E_fine, z_nm, LDOS


def compare_materials():
    """
    Compare transport in different channel materials.

    Shows how material choice (effective mass, bandgap) affects
    quantum transport characteristics.
    """
    print("\n" + "=" * 60)
    print("Material Comparison")
    print("=" * 60)

    materials = [
        MaterialType.SILICON,
        MaterialType.INGAAS,
        MaterialType.GERMANIUM,
    ]

    base_params = {
        'channel_length': 10e-9,
        'nanowire_radius': 2.5e-9,
        'nz': 30,
        'nr': 1,
        'vg': 0.3,
        'source_doping': 1e20,
        'drain_doping': 1e20,
        'channel_doping': 1e16,
        'temperature': 300,
    }

    vd_values = np.linspace(0.01, 0.25, 6)
    results_by_material = {}

    for mat_type in materials:
        mat = Material.get_material(mat_type)
        print(f"\n--- {mat.name} (m* = {mat.effective_mass} m_e) ---")

        currents = []
        for vd in vd_values:
            params = GAADeviceParams(**base_params, vd=vd)
            params.material = mat

            solver = SelfConsistentNEGF(params, mixing=0.2, max_iter=15, tol=1e-3)
            results = solver.solve(E_min=-0.4, E_max=0.6, n_energy=30, verbose=False)
            currents.append(results['current'])

        results_by_material[mat.name] = np.array(currents)

    # Plot comparison
    plt.figure(figsize=(10, 7))

    for name, currents in results_by_material.items():
        plt.plot(vd_values, currents * 1e6, 'o-', linewidth=2, markersize=8, label=name)

    plt.xlabel('Drain Voltage V_D (V)', fontsize=12)
    plt.ylabel('Drain Current I_D (µA)', fontsize=12)
    plt.title(f'Material Comparison (V_G = {base_params["vg"]} V)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('material_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results_by_material


def demonstrate_non_scf_analysis():
    """
    Quick non-self-consistent analysis for educational purposes.

    Shows the basic NEGF calculations without the Poisson equation loop.
    """
    print("\n" + "=" * 60)
    print("Non-Self-Consistent NEGF Analysis")
    print("=" * 60)

    params = GAADeviceParams(
        channel_length=10e-9,
        nanowire_radius=2.5e-9,
        nz=40,
        nr=1,
        vg=0.3,
        vd=0.1,
        temperature=300,
    )

    # Create NEGF solver without self-consistency
    negf = NEGFSolver(params)

    # Build Hamiltonian with simple linear potential drop
    z = negf.z
    potential = params.vs + (params.vd - params.vs) * z / params.channel_length

    negf.build_hamiltonian(potential)

    # Calculate transmission spectrum
    E_values = np.linspace(-0.5, 0.8, 100)
    transmission = []

    print("Calculating transmission spectrum...")
    for E in E_values:
        T = negf.calculate_transmission(E)
        transmission.append(T)

    transmission = np.array(transmission)

    # Calculate current without self-consistency
    current, spectrum = negf.calculate_current(-0.3, 0.5, 80)

    print(f"\nResults (non-self-consistent):")
    print(f"  Current: {current * 1e6:.4f} µA")
    print(f"  Peak transmission: {transmission.max():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Transmission
    axes[0].plot(E_values, transmission, 'b-', linewidth=2)
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_ylabel('Transmission T(E)')
    axes[0].set_title('Transmission Spectrum')
    axes[0].grid(True, alpha=0.3)

    # Potential profile
    axes[1].plot(z * 1e9, potential, 'r-', linewidth=2)
    axes[1].set_xlabel('Position (nm)')
    axes[1].set_ylabel('Potential (eV)')
    axes[1].set_title('Applied Potential Profile')
    axes[1].grid(True, alpha=0.3)

    # Hamiltonian visualization
    im = axes[2].imshow(np.abs(negf.H[:20, :20]), cmap='Blues')
    axes[2].set_title('|H| (first 20x20 elements)')
    axes[2].set_xlabel('j')
    axes[2].set_ylabel('i')
    plt.colorbar(im, ax=axes[2], label='|H_ij| (eV)')

    plt.tight_layout()
    plt.savefig('non_scf_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return E_values, transmission, current


if __name__ == "__main__":
    print("=" * 60)
    print("NEGF Transport Simulation Examples for GAA Transistor")
    print("=" * 60)

    # Run demonstrations
    print("\n\nSelect simulation to run:")
    print("1. Single bias point analysis")
    print("2. Transfer characteristics (Id-Vg)")
    print("3. Output characteristics (Id-Vd)")
    print("4. LDOS visualization")
    print("5. Material comparison")
    print("6. Non-self-consistent analysis (quick)")
    print("7. Run all")

    try:
        choice = input("\nEnter choice (1-7) [default: 1]: ").strip()
        if not choice:
            choice = "1"
        choice = int(choice)
    except (ValueError, EOFError):
        choice = 1
        print("Running default: Single point analysis")

    if choice == 1 or choice == 7:
        single_point_analysis()

    if choice == 2 or choice == 7:
        calculate_transfer_characteristics()

    if choice == 3 or choice == 7:
        calculate_output_characteristics()

    if choice == 4 or choice == 7:
        demonstrate_ldos()

    if choice == 5 or choice == 7:
        compare_materials()

    if choice == 6 or choice == 7:
        demonstrate_non_scf_analysis()

    print("\n" + "=" * 60)
    print("Simulations complete!")
    print("=" * 60)
