# Q2 Schrodinger Equation Solver

This folder contains a finite-difference solution for 個人作業 Q2.

Run:

```bash
python3 self/schrodinger_q2.py
```

Default model:

- finite square quantum well
- effective mass `m* = 0.067 m0`
- well width `10 nm`
- barrier height `V0 = 0.30 eV`
- calculation window `z = -20 ... 20 nm`

The defaults are adjusted to reproduce the three bound-state energy levels and
wavefunction shapes in the assignment reference figure. Outputs are written to
`self/output/`:

- `q2_quantum_well_overlay.png`: potential, energy levels, and wavefunctions
- `q2_state_n1.png`, `q2_state_n2.png`, `q2_state_n3.png`: the three state plots
- `q2_bound_state_summary.csv`: calculated eigenenergies

You can change parameters from the command line, for example:

```bash
python3 self/schrodinger_q2.py --well-width-nm 9.5 --barrier-ev 0.32
```
