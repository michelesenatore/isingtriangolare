# Ising su reticolo triangolare

Questa è la versione 0 del mio ising, breve recap per chi lo userà
Le dipendenze python sono racchiuse dentro il file requirements

Simulazione montecarlo di un modello di Ising su reticolo triangolare.
Dentro ising_byme.py sono definite le classi del codice: 
- Tipo di reticolo (per ora solo triangolare, ma presto arriverà anche square lattice)
- Tipo di modello (per ora solamente ising)
- Specifiche del montecarlo. Per ora è un Metropolis a cui vengono passati in input gli step di equilibrio,simulazione e misura

Dentro mc_metropolis_ising avviene la vera simulazione
