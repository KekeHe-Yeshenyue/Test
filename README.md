# NEGF Transport Simulation & Sweep Data Plotter

Quantum transport simulation for Gate-All-Around (GAA) transistors using the Non-Equilibrium Green's Function (NEGF) method, along with a plotting tool for sweep data visualization.

## Project Structure

| File | Description |
|------|-------------|
| `negf_transport.py` | Core NEGF solver: Hamiltonian construction, Green's functions, transmission, and self-consistent Poisson–NEGF loop |
| `run_gaa_simulation.py` | Example scripts for single-point analysis, transfer/output characteristics, LDOS, and material comparison |
| `test_negf.py` | Unit tests for the NEGF transport module |
| `plot_sweep.py` | Standalone script to plot line curves from sweep data text files |
| `test SWEEP.txt` | Sample voltage sweep data (V_G vs I_D) |
| `Claude test SWEEP.txt` | Copy of sample sweep data for quick reference |

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Plotting Sweep Data

`plot_sweep.py` reads a tab/space/comma-delimited text file and plots the line curve. It auto-detects column headers and applies both linear and log scale when the data spans multiple orders of magnitude.

### Usage

```bash
# Use the default data path (~/Desktop/Data needed to be processed/test SWEEP.txt)
python plot_sweep.py

# Specify a custom file
python plot_sweep.py /path/to/your/data.txt
```

### Expected Data Format

```
# Comment or description line
# Column1_Label	Column2_Label
-0.20	1.23e-12
-0.10	3.45e-11
0.00	8.76e-10
...
```

- Lines starting with `#` are treated as comments or headers.
- Data columns can be separated by tabs, spaces, or commas.
- Scientific notation (e.g. `1.23e-12`) is supported.

The output plot is saved as a `.png` file in the same directory as the input data.

## Running Simulations

```bash
python run_gaa_simulation.py
```

Choose from the interactive menu:
1. Single bias point analysis
2. Transfer characteristics (I_D vs V_G)
3. Output characteristics (I_D vs V_D)
4. LDOS visualization
5. Material comparison
6. Non-self-consistent analysis
7. Run all

## Running Tests

```bash
python test_negf.py
```
