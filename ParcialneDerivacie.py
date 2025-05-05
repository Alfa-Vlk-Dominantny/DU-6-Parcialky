import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RungeKutta4Order(f, y0, t, args=()):
    """moja funkcia z predoslej DU na ODR """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2, t[i] + h / 2, *args)
        k3 = f(y[i] + k2 * h / 2, t[i] + h / 2, *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
    return y


def laplace_operator(phi, h):
    """Diskrétny Laplaceov operátor -  je to numericka aproximácia laplaca, resp. druhej derivacia"""
    laplace = np.zeros_like(phi)
    laplace[1:-1, 1:-1] = (phi[1:-1, 2:] + phi[1:-1, :-2] +
                           phi[2:, 1:-1] + phi[:-2, 1:-1] - 4 * phi[1:-1, 1:-1]) / h ** 2

    #PS Okraje
    laplace[0,1:-1] = (phi[0,2:] + phi[0,:-2] + phi[1,1:-1] - 3 * phi[0,1:-1]) / h ** 2
    laplace[-1,1:-1] = (phi[-1,2:] + phi[-1,:-2] + phi[-2,1:-1] - 3 * phi[-1,1:-1]) / h ** 2
    laplace[1:-1,0] = (phi[2:,0] + phi[:-2,0] + phi[1:-1,1] -  3 * phi[1:-1,0]) / h ** 2
    laplace[1:-1,-1] = (phi[2:,-1] + phi[:-2,-1] + phi[1:-1,-2] - 3 * phi[1:-1,-1]) / h ** 2

    #PS Rohy
    laplace[0,0] = (phi[0,1] + phi[1,0] - 2 * phi[0,0]) / h ** 2
    laplace[0,-1] = (phi[0,-2] + phi[1,-1] - 2 * phi[0,-1]) / h ** 2
    laplace[-1,0] = (phi[-1,1] + phi[-2,0] - 2 * phi[-1,0]) / h ** 2
    laplace[-1,-1] = (phi[-1,-2] + phi[-2,-1] - 2 * phi[-1,-1]) / h ** 2

    return laplace


def laplace_equation(phi_flat, t, h, fixed_flat, shape):
    """Wrapper pre RK4, čiže pretransformuje nám vstup na 1D vektor nech ho vieme v RK4 pouzit"""
    phi = phi_flat.reshape(shape)
    fixed = fixed_flat.reshape(shape)
    dphi_dt = laplace_operator(phi, h)
    dphi_dt[fixed] = 0
    return dphi_dt.flatten()


"""Nejake parametre kondenzatoru u vsetkych uloh"""
nx, ny = 50, 50
h = 1.0 / (nx - 1)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
U = 1.0
t_max = 10.0
dt = 0.1 * h ** 2
t = np.arange(0, t_max, dt)

#########################################
# Uloha 1: Kondenzátor
#########################################

phi1 = np.zeros((ny, nx))
phi1[0, :] = 0
phi1[-1, :] = U
fixed_mask1 = np.zeros_like(phi1, dtype=bool)
fixed_mask1[0, :] = fixed_mask1[-1, :] = True

"""Nastavili sme okrajové podmienky"""

solution1 = RungeKutta4Order(
    lambda phi, t: laplace_equation(phi, t, h, fixed_mask1.flatten(), phi1.shape),
    phi1.flatten(), t
)
phi1_final = solution1[-1].reshape(phi1.shape)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, phi1_final, cmap='viridis')
ax1.set_title('Uloha 1: Kondenzátor (3D)')

ax2 = fig.add_subplot(122)
cf = ax2.contourf(X, Y, phi1_final, levels=20, cmap='viridis')
plt.colorbar(cf, label='Potenciál φ')
ax2.set_title('Uloha 1: Vrstevnice')
plt.tight_layout()
plt.show()
print()
#########################################
# Uloha 2: Búrka
#########################################

phi2 = np.zeros((ny, nx))
phi2[0, :] = 0
phi2[-1, :] = U

person_width, person_height = 0.1, 0.5
person_x_center = 0.5
person_mask = ((X >= person_x_center - person_width / 2) &
               (X <= person_x_center + person_width / 2) &
               (Y <= person_height))
phi2[person_mask] = 0
fixed_mask2 = np.zeros_like(phi2, dtype=bool)
fixed_mask2[0, :] = fixed_mask2[-1, :] = True
fixed_mask2[person_mask] = True

solution2 = RungeKutta4Order(
    lambda phi, t: laplace_equation(phi, t, h, fixed_mask2.flatten(), phi2.shape),
    phi2.flatten(), t
)
phi2_final = solution2[-1].reshape(phi2.shape)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, phi2_final, cmap='viridis')
ax1.set_title('Uloha 2: Mrak a osoba (3D)')

ax2 = fig.add_subplot(122)
cf = ax2.contourf(X, Y, phi2_final, levels=20, cmap='viridis')
ax2.fill(
    [person_x_center - person_width / 2, person_x_center + person_width / 2,
     person_x_center + person_width / 2, person_x_center - person_width / 2],
    [0, 0, person_height, person_height],
    color='gray', alpha=0.5, label='Osoba'
)
plt.colorbar(cf, label='Potenciál φ')
ax2.legend()
ax2.set_title('Uloha 2: Vrstevnice s osobou')
plt.tight_layout()
plt.show()

"""
To je inak dosť zaujímavé interpretovať dôsledky z tejto ulohy, keďže mne ešte velmi nejde elmag. Tým, že my ako osoba sme (v idealnom pripade) homogenny vodic, tak ekvipotenciály
sa deformujú a ohýbajú okolo nás, čím dôjde k zhusteniu ekvipotenciál nad nami a výraznému zvýšeniu gradientu potenciálu a pokiaľ sa ionizuje vzduch a vytvorí blesk, tak hladá cestu s najmenším
odporom, resp najvyšším gradientom, preto je to pre nás nebezpečné a môžeme umrieť.
"""
print()

#########################################
# Uloha 3: Naboj
#########################################


epsilon_0 = 8.854e-12

""" Poisson """
rho = -epsilon_0 * laplace_operator(phi2_final, h)

head_mask = ((X >= 0.5 - person_width / 2) & (X <= 0.5 + person_width / 2) &
             (Y >= person_height * 0.8) & (Y <= person_height))
head_charge_density = rho[head_mask].mean()

print(f"Priemerná hustota náboja na hlave: {head_charge_density:.2e} C/m²")

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contourf(X, Y, rho, levels=50, cmap='coolwarm')
plt.colorbar(label='Hustota náboja [C/m²]')
plt.title('Rozloženie hustoty náboja')
plt.fill([0.45, 0.55, 0.55, 0.45], [0.4, 0.4, 0.5, 0.5], 'gray', alpha=0.5)  # Hlava

plt.subplot(122)
plt.contourf(X, Y, phi2_final, levels=20, cmap='viridis')
plt.colorbar(label='Potenciál [V]')
plt.title('Potenciálové pole')
plt.tight_layout()
plt.show()
print()

#########################################
# Uloha 4: Hromozvod
#########################################

# Parametre hromozvodu
#PS Tady je vidět, jak jsou někdy komentáře na škodu. Komentář je o osobě, přitom ale definujete hromosvod.
lightning_rod_height = 0.8  # Nasa osoba je vysoka 0,5
rod_width = 0.02
rod_x_center = 0.7  #Posunuly sme doprava
rod_mask = ((X >= rod_x_center - rod_width / 2) &
            (X <= rod_x_center + rod_width / 2) &
            (Y <= lightning_rod_height))


phi4 = np.zeros((ny, nx))
phi4[0, :] = 0      # zem
phi4[-1, :] = U      # mrak
phi4[person_mask] = 0  # osoba má potenciál zeme
phi4[rod_mask] = 0     # hromosvod má tiež potenciál zeme


fixed_mask4 = np.zeros_like(phi4, dtype=bool)
fixed_mask4[0, :] = fixed_mask4[-1, :] = True  # okraje
fixed_mask4[person_mask] = True                # osoba
fixed_mask4[rod_mask] = True                   # hromosvod


solution4 = RungeKutta4Order(
    lambda phi, t: laplace_equation(phi, t, h, fixed_mask4.flatten(), phi4.shape),
    phi4.flatten(), t
)
phi4_final = solution4[-1].reshape(phi4.shape)


rho_with_rod = -epsilon_0 * laplace_operator(phi4_final, h)
head_charge_with_rod = rho_with_rod[head_mask].mean()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


cont1 = axes[0].contourf(X, Y, phi4_final, levels=20, cmap='viridis')
fig.colorbar(cont1, ax=axes[0], label='Potenciál [V]')

axes[0].fill_between([person_x_center - person_width/2, person_x_center + person_width/2],
                     0, person_height, color='gray', alpha=0.5, label='Osoba')

axes[0].fill_between([rod_x_center - rod_width/2, rod_x_center + rod_width/2],
                     0, lightning_rod_height, color='red', alpha=0.5, label='Hromosvod')
axes[0].set_title('Potenciálové pole s hromozvodom')
axes[0].legend()


axes[1].bar(['Bez hromozvodu', 'S hromozvodom'],
            [abs(head_charge_density), abs(head_charge_with_rod)],
            color=['blue', 'green'])
axes[1].set_ylabel('Hustota náboja [C/m²]')
axes[1].set_title('Porovnanie náboja na hlave')


cont3 = axes[2].contourf(X, Y, rho_with_rod, levels=50, cmap='coolwarm')
fig.colorbar(cont3, ax=axes[2], label='Hustota náboja [C/m²]')

axes[2].fill_between([person_x_center - person_width/2, person_x_center + person_width/2],
                     0, person_height, color='gray', alpha=0.3)
axes[2].fill_between([rod_x_center - rod_width/2, rod_x_center + rod_width/2],
                     0, lightning_rod_height, color='red', alpha=0.3)
axes[2].set_title('Rozloženie náboja s hromozvodom')

plt.tight_layout()
plt.show()

"""
Mám ale otázku čo som videl spustenie tohoto pola. prečo by blesk udrel do nás? ano, hlada cestu najmenšieho odporu ale tá je daná jednoznačne, tak prečo
by blesk mohol skočiť na nás? Akože viem , že hromozvod znižuje pravdepodobnosť.
"""
print(f"Hustota náboja na hlave s hromozvodom: {head_charge_with_rod:.2e} C/m²")
print(f"Pokles náboja: {100 * (head_charge_density - head_charge_with_rod) / head_charge_density:.1f}%")

#########################################
# Uloha 5: Polia
#########################################

def calculate_electric_field(phi, h):
    """Vypočíta E = -grad(φ) pomocou centrálnych diferencií"""
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)

    # parcialka podla x
    Ex[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * h)

    # parcialka podla y
    Ey[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * h)

    return Ex, Ey


fig, axes = plt.subplots(3, 2, figsize=(15, 18))

Ex1, Ey1 = calculate_electric_field(phi1_final, h)
E_mag1 = np.sqrt(Ex1 ** 2 + Ey1 ** 2)

cont1 = axes[0, 0].contourf(X, Y, phi1_final, levels=20, cmap='viridis')
plt.colorbar(cont1, ax=axes[0, 0], label='Potenciál [V]')
axes[0, 0].set_title('1. Kondenzátor: Potenciál')

strm1 = axes[0, 1].streamplot(X, Y, Ex1, Ey1, color=E_mag1, cmap='plasma', density=1.5)
plt.colorbar(strm1.lines, ax=axes[0, 1], label='|E| [V/m]')
axes[0, 1].set_title('1. Kondenzátor: Intenzita poľa')

Ex2, Ey2 = calculate_electric_field(phi2_final, h)
E_mag2 = np.sqrt(Ex2 ** 2 + Ey2 ** 2)

cont2 = axes[1, 0].contourf(X, Y, phi2_final, levels=20, cmap='viridis')
plt.colorbar(cont2, ax=axes[1, 0], label='Potenciál [V]')
#PS Pro oblasti je i tady dobré použít parametry polohy osoby a hromosvodu, jako máte výše.
#PS Vše bude dobře fungovat i v případě, že změníte polohu nebo velikost osoby.
axes[1, 0].fill_between([0.45, 0.55], 0, 0.5, color='gray', alpha=0.5, label='Osoba')
axes[1, 0].set_title('2. Osoba: Potenciál')

strm2 = axes[1, 1].streamplot(X, Y, Ex2, Ey2, color=E_mag2, cmap='plasma', density=1.5)
plt.colorbar(strm2.lines, ax=axes[1, 1], label='|E| [V/m]')
axes[1, 1].fill_between([0.45, 0.55], 0, 0.5, color='gray', alpha=0.5)
axes[1, 1].set_title('2. Osoba: Intenzita poľa')

Ex3, Ey3 = calculate_electric_field(phi4_final, h)
E_mag3 = np.sqrt(Ex3 ** 2 + Ey3 ** 2)

cont3 = axes[2, 0].contourf(X, Y, phi4_final, levels=20, cmap='viridis')
plt.colorbar(cont3, ax=axes[2, 0], label='Potenciál [V]')
axes[2, 0].fill_between([0.45, 0.55], 0, 0.5, color='gray', alpha=0.5, label='Osoba')
axes[2, 0].fill_between([0.49, 0.51], 0, 0.7, color='red', alpha=0.5, label='Hromosvod')
axes[2, 0].legend()
axes[2, 0].set_title('3. S hromosvodom: Potenciál')

strm3 = axes[2, 1].streamplot(X, Y, Ex3, Ey3, color=E_mag3, cmap='plasma', density=1.5)
plt.colorbar(strm3.lines, ax=axes[2, 1], label='|E| [V/m]')
axes[2, 1].fill_between([0.45, 0.55], 0, 0.5, color='gray', alpha=0.5)
axes[2, 1].fill_between([0.49, 0.51], 0, 0.7, color='red', alpha=0.5)
axes[2, 1].set_title('3. S hromosvodom: Intenzita poľa')

plt.tight_layout()
plt.show()

print("\nMaximálne intenzity poľa:")
print(f"1. Kondenzátor: {E_mag1.max():.2f} V/m")
print(f"2. Osoba: {E_mag2.max():.2f} V/m (zmena: {(E_mag2.max() - E_mag1.max()) / E_mag1.max() * 100:.1f}%)")
print(f"3. S hromosvodom: {E_mag3.max():.2f} V/m (zmena: {(E_mag3.max() - E_mag1.max()) / E_mag1.max() * 100:.1f}%)")

print()