import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# Constants
g = 9.81  # gravity [m/s^2]
P_atm = 101325  # atmospheric pressure [Pa]
rho_w = 1000  # density of water [kg/m^3]
eta = 0.001  # dynamic viscosity of water [Pa·s]
m = 0.000001  # mass of bubble [kg]
R = 8.314  # gas constant [J/(mol·K)]
T = 298  # temperature [K]
M = 0.029  # molar mass of air [kg/mol]
Cd = 0.47 # drag coefficient for a sphere

# Defining radius of a bubble depending on depth
def r(h):
    return np.cbrt((3 * R * m * T) / (4 * np.pi * M * (rho_w * g * (-h) + P_atm))) # h is depth on y axis (h < 0), so -h is used to obtain positive pressure

# Defining Buoyancy force depending on depth 
def buoyancy(h):
    return rho_w * g * (4 / 3) * np.pi * r(h) ** 3 

# Defining Stokes force depending on depth and velocity (negative, opposite direction to the velocity)
def stokes(h, v):
    return - 6 * np.pi * v * eta * r(h)  

# Defining drag force depending on depth and velocity (negative, opposite direction to the velocity)
def drag_v2(h, v):
    return - 0.5 * Cd * np.pi * rho_w *  r(h) ** 2 * v ** 2

# Function defining the ODEs
def f(t, Y):
    h = Y[0]
    v = Y[1]
    return np.hstack((v, (buoyancy(h) + stokes(h, v) + drag_v2(h, v)) / m - g))

# Condition to stop simulation when the air bubble touches surface
def con_s(t, Y):
    return Y[0] - h_max

con_s.terminal = True

# Initial state parameters
h_max = -r(0)  # bubble must be submerged
h0 = -10  # [m]
v0 = 0 #0.01  # [m/s]
Y0 = np.hstack((h0, v0))
t0 = 0
te = 2 # [s]
max_ts = 0.0001 # [s]

# Solving the ODEs and extracting solutions
sol = solve_ivp(f, (t0, te), Y0, max_step=max_ts, events=con_s, atol=1e-12, rtol=1e-12)
Y = sol.y
t = sol.t
h = Y[0]
v = Y[1]

# Checking the results with energy conservation
dr = np.diff(h)

# Calculating midpoints of values for calculations of force in each segment
drm = (h[1:] + h[:-1]) / 2 
dv = (v[1:] + v[:-1]) / 2

# Calculation of energy balance 
wb = np.hstack((0, np.cumsum(buoyancy(drm) * dr)))
ws = np.hstack((0, np.cumsum(stokes(drm, dv) * dr)))
wd = np.hstack((0, np.cumsum(drag_v2(drm, dv) * dr)))

w = wb + ws + wd  # sum of work done

KE = m * v ** 2 / 2  # kinetic energy
PE = m * g * h  # potential energy

E_t = KE + PE  # total energy
E = E_t - w  # check if total energy is equal to work done
E0 = E[0]

E_cons = (E - E0) / E0 # relative change

# Plotting all results 

fig, axs = plt.subplots(2, 3, figsize=(16, 8)) 

# Plotting Depth Over Time
axs[0, 0].plot(t, h, linestyle="None", marker="o", markersize=1, color="crimson")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Depth (m)")
axs[0, 0].set_title("Bubble Depth Over Time")

# Plotting Velocity Over Time
axs[0, 1].plot(t, v, linestyle="None", marker="o", markersize=1, color="crimson")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Velocity (m/s)")
axs[0, 1].set_title("Bubble Velocity Over Time")

# Plotting Kinetic Energy Over Time
axs[0, 2].plot(t, KE, linestyle="None", marker="o", markersize=1, color="navy")
axs[0, 2].set_xlabel("Time (s)")
axs[0, 2].set_ylabel("Kinetic Energy (J)")
axs[0, 2].set_title("Kinetic Energy Over Time")

# Plotting Work of Each Force Over Time
axs[1, 0].plot(t, wb, linestyle="None", marker="o", markersize=1, label='Buoyancy Work', color="darkblue")
axs[1, 0].plot(t, ws, linestyle="None", marker="o", markersize=1, label='Stokes Work', color="purple")
axs[1, 0].plot(t, wd, linestyle="None", marker="o", markersize=1, label='Drag Work', color="mediumpurple")
axs[1, 0].plot(t, w, linestyle="None", marker="o", markersize=1, label='Sum of Work', color="darkorange")
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Work (J)')
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_title("Work of Forces Over Time")

# Plotting Energy Over Time
axs[1, 1].plot(t, KE, linestyle="None", marker="o", markersize=1, label='Kinetic Energy', color="navy")
axs[1, 1].plot(t, PE, linestyle="None", marker="o", markersize=1, label='Potential Energy', color="mediumpurple")
axs[1, 1].plot(t, E, linestyle="None", marker="o", markersize=1, label='Sum of Energy', color="darkorange")
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Energy (J)')
axs[1, 1].legend()
axs[1, 1].grid()
axs[1, 1].set_title("Energy Over Time")

# Plotting Relative Energy Change Over Time
axs[1, 2].plot(t, E_cons, linestyle="None", marker="o", markersize=1, color="navy")  
axs[1, 2].set_xlabel("Time (s)")
axs[1, 2].set_ylabel("Relative Energy Change (ΔE / E0)")
axs[1, 2].grid()
axs[1, 2].set_title("Energy Conservation Check ")
plt.tight_layout()
plt.show()

# An animation representing the behaviour of air bubble.

radii = r(h)
t_uni = np.linspace(t[0], t[-1], 1000)
h_uni = interp1d(t, h, kind='cubic')(t_uni)
v_uni = interp1d(t, v, kind='cubic')(t_uni)
radii_uni = interp1d(t, radii, kind='cubic')(t_uni)

fig, ax = plt.subplots(figsize=(6, 8))
bubble, = ax.plot([], [], 'o', color='blue', alpha=0.8)

info_text = ax.text(
    0.05, 0.05, "", transform=ax.transAxes, fontsize=10, verticalalignment='bottom'
)

ax.set_xlim(-1, 1)
ax.set_ylim(h0, 0)
ax.set_ylabel("Depth (m)")
ax.set_title("Bubble Rising in Water")

def animate(i):
    bubble.set_data([0], [h_uni[i]])  
    bubble.set_markersize(radii_uni[i] * 10000) 

    info_text.set_text(
        f"Depth (h): {h_uni[i]:.2f} m\n"
        f"Velocity (v): {v_uni[i]:.2f} m/s\n"
        f"Radius (r): {radii_uni[i]:.2e} m"
    )
    return bubble, info_text

interval = (t_uni[-1] / len(t_uni)) * 1000  

ani = FuncAnimation(fig, animate, frames=len(t_uni), interval=interval, blit=True)

plt.show()
