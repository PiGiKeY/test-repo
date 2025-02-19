import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Definicja transmitancji: H(s) = (s + 1) / (s^2 + 2s + 1)
numerator = [1, 1]  # Licznik
denominator = [1, 2, 5]  # Mianownik

# Tworzenie obiektu transmitancji
system = signal.TransferFunction(numerator, denominator)

# Analiza odpowiedzi skokowej
t, response = signal.step(system)

# Wykres odpowiedzi skokowej
plt.plot(t, response)
plt.title("Odpowiedź skokowa systemu")
plt.xlabel("Czas [s]")
plt.ylabel("Odpowiedź")
plt.grid()
plt.show()
