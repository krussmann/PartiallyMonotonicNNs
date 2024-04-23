import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
csv_datei = "monotonic_predictions.csv"
daten = pd.read_csv(csv_datei)


csv_x = "test_data.csv"

daten_x = pd.read_csv(csv_x)

# Spalten auswählen, die geplottet werden sollen (z.B. Spalte 'x' und 'y')
x_werte = daten_x["x"]
y_werte = daten_x["y"]
orig_werte = daten['# f_x']
y_werte_upper = daten[' upper envelope']
y_werte_lower = daten[' lower envelope']

# Plot erstellen
plt.figure(figsize=(10, 6))  # Optional: Größe des Plots festlegen
plt.plot(x_werte[:-1], orig_werte, linestyle='-', label="orig")  # Du kannst den Plot anpassen (Linienstil, Marker, etc.)
plt.plot(x_werte[:-1], y_werte[:-1], linestyle='-', label="orig_txt")  # Du kannst den Plot anpassen (Linienstil, Marker, etc.)
plt.plot(x_werte[:-1], y_werte_upper, linestyle='-', label="Upper")  # Du kannst den Plot anpassen (Linienstil, Marker, etc.)
plt.plot(x_werte[:-1], y_werte_lower, linestyle='-', label="Lower")  # Du kannst den Plot anpassen (Linienstil, Marker, etc.)

# Achsentitel und Titel hinzufügen
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('CSV-Datenplot')
plt.legend()

# Plot anzeigen
plt.grid(True)  # Optional: Gitterlinien hinzufügen
plt.show()
