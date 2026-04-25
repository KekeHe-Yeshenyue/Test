#!/usr/bin/env python3
"""
Plot line curve from sweep data in a text file.

Reads tab/space/comma-delimited numeric data from a .txt file,
auto-detects headers and columns, and plots the line curve.

Usage:
    python plot_sweep.py
    python plot_sweep.py /path/to/your/data.txt
"""

import os
import sys

if sys.version_info[0] < 3:
    sys.stderr.write(
        "ERROR: This script requires Python 3.5 or newer. "
        "You are running Python {}.{}.\n".format(sys.version_info[0], sys.version_info[1])
    )
    sys.exit(1)

import platform
import numpy as np
import matplotlib
if platform.system() == 'Darwin':
    for backend in ['macosx', 'TkAgg', 'Qt5Agg']:
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt


def find_default_data_file():
    """Locate the default data file on Desktop."""
    # Hard-coded path for the user's macOS machine
    mac_path = "/Users/bmy/Desktop/Data needed to be processed/test SWEEP.txt"
    if os.path.isfile(mac_path):
        return mac_path

    home = os.path.expanduser("~")
    default = os.path.join(home, "Desktop", "Data needed to be processed", "test SWEEP.txt")
    if os.path.isfile(default):
        return default

    # Fallback: search common home directories
    for base in ["/Users/bmy", "/home/user", os.path.expanduser("~")]:
        candidate = os.path.join(base, "Desktop", "Data needed to be processed", "test SWEEP.txt")
        if os.path.isfile(candidate):
            return candidate
    return mac_path


def parse_sweep_file(filepath):
    """
    Parse a sweep data text file.

    Handles:
    - Comment lines starting with #
    - Tab, space, or comma delimiters
    - Header lines with column names
    - Numeric data in scientific notation

    Returns:
        headers: list of column name strings (or None)
        data: numpy array of shape (n_rows, n_cols)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Data file not found: {}".format(filepath))

    headers = None
    data_lines = []

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for comment/header lines
            if stripped.startswith('#'):
                # Try to extract column headers from the last comment line
                content = stripped.lstrip('#').strip()
                # If it looks like column headers (contains tab or multiple words
                # that could be labels), save it
                if '\t' in content or content.count('(') >= 2 or content.count(',') >= 1:
                    # Split by tab, comma, or multiple spaces
                    for delimiter in ['\t', ',']:
                        if delimiter in content:
                            headers = [h.strip() for h in content.split(delimiter)]
                            break
                    else:
                        headers = content.split()
                continue

            # Try to parse as numeric data
            # Detect delimiter
            for delimiter in ['\t', ',', None]:  # None means whitespace
                try:
                    if delimiter is not None:
                        values = [float(x.strip()) for x in stripped.split(delimiter) if x.strip()]
                    else:
                        values = [float(x) for x in stripped.split()]
                    if len(values) >= 2:
                        data_lines.append(values)
                        break
                except ValueError:
                    continue

    if not data_lines:
        raise ValueError("No numeric data found in {}".format(filepath))

    # Ensure all rows have the same number of columns
    n_cols = len(data_lines[0])
    data_lines = [row for row in data_lines if len(row) == n_cols]

    data = np.array(data_lines)
    return headers, data


def plot_sweep(filepath):
    """Read sweep data and plot the line curve."""
    print("Reading data from: {}".format(filepath))
    headers, data = parse_sweep_file(filepath)

    n_rows, n_cols = data.shape
    print("Loaded {} data points with {} columns".format(n_rows, n_cols))

    x = data[:, 0]
    y = data[:, 1]

    # Determine axis labels
    if headers and len(headers) >= 2:
        xlabel = headers[0]
        ylabel = headers[1]
    else:
        xlabel = "X"
        ylabel = "Y"

    # Check if y data spans multiple orders of magnitude (common for I-V sweeps)
    y_abs = np.abs(y)
    y_positive = y_abs[y_abs > 0]
    use_log = len(y_positive) > 1 and (y_positive.max() / y_positive.min()) > 1e3

    if use_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Linear scale plot
        ax1.plot(x, y, 'b-o', linewidth=2, markersize=5)
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title("Sweep Curve (Linear Scale)", fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Log scale plot
        ax2.semilogy(x, y_abs, 'r-o', linewidth=2, markersize=5)
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel("|{}|".format(ylabel), fontsize=12)
        ax2.set_title("Sweep Curve (Log Scale)", fontsize=13)
        ax2.grid(True, which='both', alpha=0.3)

        plt.suptitle(os.path.basename(filepath), fontsize=14, fontweight='bold')
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))

        ax1.plot(x, y, 'b-o', linewidth=2, markersize=5)
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title(os.path.basename(filepath), fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # If there are more columns, plot them too
    if n_cols > 2:
        for col_idx in range(2, n_cols):
            label = headers[col_idx] if headers and len(headers) > col_idx else "Column {}".format(col_idx + 1)
            ax1.plot(x, data[:, col_idx], '-o', linewidth=2, markersize=5, label=label)
        ax1.legend()

    plt.tight_layout()

    # Save the figure
    output_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, "{}_plot.png".format(base_name))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("Plot saved to: {}".format(output_path))

    plt.show(block=True)
    print("Plot window closed.")
    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = find_default_data_file()

    plot_sweep(file_path)
