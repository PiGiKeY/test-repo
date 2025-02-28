import control as ctrl
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#Definicja transmitancji: H(s) = (s + 1) / (s^2 + 2s + 1)
numerator = [1]  # Licznik
denominator = [1, 6, 5, 0]  # Mianownik

#Tworzenie obiektu transmitancji
plant = ctrl.TransferFunction(numerator, denominator)

# Definicja parametrów PID wg metody Nicholsa

Ku=30

# Tworzenie obiektu transmitancji PID
pid_numerator = [Ku]
pid_denominator = [1]
pidZN = ctrl.TransferFunction(pid_numerator, pid_denominator)

systemZN = pidZN*plant

#t, response_pid = signal.step(system,T=time)

closed_loopZN=ctrl.feedback(systemZN,1)
time = np.linspace(0, 20, 2000)
t, responseZN = ctrl.step_response(closed_loopZN, time)


# Wykrywanie szczytów w odpowiedzi
peaks, _ = signal.find_peaks(responseZN)
peak_times = t[peaks]

# Obliczanie okresu oscylacji
if len(peak_times) > 1:
    Tu = np.mean(np.diff(peak_times))
    print(f"Ultimate period (Tu): {Tu} seconds")
else:
    print("Not enough peaks detected to determine the ultimate period.")
    Tu = 1

# Calculate PID gains using Ziegler-Nichols method
Kp = 0.6 * Ku
Ki = 1.2 * Ku / Tu
Kd = 0.075 * Ku * Tu

pid = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
# Define the open-loop transfer function
open_loop = pid * plant

t, response = ctrl.step_response(plant, time)

# Define the closed-loop transfer function
closed_loop = ctrl.feedback(open_loop, 1)
time = np.linspace(0, 20, 2000)
t, response_pid = ctrl.step_response(closed_loop, time)

#Wykres odpowiedzi skokowej
plt.plot(t, response, label="Odpowiedź bez PID")
plt.plot(t, response_pid, label="Odpowiedź z PID", linestyle='--')
plt.title("Odpowiedź skokowa systemu z szumem")
plt.xlabel("Czas [s]")
plt.ylabel("Odpowiedź")
plt.legend()
plt.grid()
plt.show()
