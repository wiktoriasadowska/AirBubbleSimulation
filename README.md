# Air Bubble Simulation

This Python project simulates the vertical movement of an air bubble rising in water, considering physical forces such as buoyancy, drag, and Stokes resistance. The simulation also tracks energy conservation over time and provides data visualization and animation of the process.

## Physics modeled:
- **Buoyancy force** (depending on depth)
- **Viscous (Stokes) drag** (linear with velocity)
- **Quadratic drag** (depending on velocity squared)
- **Variable bubble radius** (based on depth and pressure using the ideal gas law)

## Features:
- Solves ODEs using `solve_ivp` (SciPy)
- Tracks depth, velocity, kinetic and potential energy
- Calculates and plots energy balance and work done by forces
- Generates an animated visualization of the bubble rising

## Technologies used:
- Python
- NumPy
- SciPy
- Matplotlib
- FuncAnimation


