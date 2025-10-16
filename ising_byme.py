import numpy as np
from typing import Optional, Dict, Any
from f2py_jit import jit
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
f90 = jit('mc_metropolis_ising.f90', flags='-O3')  # evita -ffast-math se vuoi riproducibilità


#@dataclass
#class SquareLattice:
#    L :int
#    a:np.float32
#    NN_vectors = np.array([1,0],[-1,0],[0,1],[0.-1])
#    def __post_init__(self):
#        self.N = self.L * self.L
#        self.positions = self._positions()
#        self.neighbour = self._neighbour()
#    def _positions(self):
#        """
#        Genero tutte le posizioni R del reticolo di bravais quadrato piano, a partire da nx e ny che ne definiscono l'estensione.
#        """
#        positions = np.empty((self.N,2),dtype=np.float32)
#        for j in range(self.L):
#            for i in range(self.L):
#                idx = j*self.L + i
#                positions[idx,0] = i + j * self.a
#                positions[idx,1] = i + j * self.a
#        return positions
#    def _neighbour(self):
#        """
#        Visualizza i vicini (primi per ora) di un qualsiasi punto appartenente al bravais triangolare.
#        Per ora lavoro con le liste
#        """
#        neighbour_list = np.empty((self.N,6),dtype=np.int32)
#        for j in range(self.ny):
#            for i in range(self.nx):
#                idx = j*self.nx + i
#                lst = []
#                for dx, dy in self.NN_vectors:
#                    ni = (i + dx) % self.nx   # PBC
#                    nj = (j + dy) % self.ny
#                    lst.append(nj*self.nx + ni)
#                neighbour_list[idx] = lst
#        return neighbour_list
@dataclass
class TriangularLattice:
    nx:int
    ny:int
    a:np.float32 = 32
    NN_vectors = np.array([[ 1, 0], [-1, 0], [ 0, 1], [ 0,-1], [-1, 1], [ 1,-1]], dtype=np.int32)

    def __post_init__(self):
        self.N = self.nx * self.ny
        self.positions = self._positions()
        self.neighbour = self._neighbour()
    def _positions(self):
        """
        Genero tutte le posizioni R del reticolo di bravais triangolare piano, a partire da nx e ny che ne definiscono l'estensione.
        """
        positions = np.empty((self.N,2),dtype=np.float32)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j*self.nx + i
                positions[idx,0] = (i + 0.5*j)*self.a
                positions[idx,1] = j * self.a * (np.sqrt(3)/2)
        return positions
    def _neighbour(self):
        """
        Visualizza i vicini (primi per ora) di un qualsiasi punto appartenente al bravais triangolare.
        Per ora lavoro con le liste
        """
        neighbour_list = np.empty((self.N,6),dtype=np.int32)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = j*self.nx + i
                lst = []
                for dx, dy in self.NN_vectors:
                    ni = (i + dx) % self.nx   # PBC
                    nj = (j + dy) % self.ny
                    lst.append(nj*self.nx + ni)
                neighbour_list[idx] = lst
        return neighbour_list
@dataclass
class IsingTriangular:  
    def __init__(self,lattice:TriangularLattice,J = np.float64(1), T =np.float64(1),init=None,rng = None):
        """
        Modello di Ising su reticolo triangolare: \\
        i parametri di input sono:\\
            J = +-1 (ferro/antiferro) \\
            T = Temperatura \\
        Viene generata una configurazione iniziale di spin, di default casuale ma può essere imposta anche ordinata.\\
        """
        self.lattice = lattice
        self.J = J
        self.T = T
        self.rng = rng or np.random.default_rng()

        if init == 'up':
            self.spins = np.ones(self.lattice.N, dtype = np.int32)
        elif init == 'down':
            self.spins = -np.ones(self.lattice.N, dtype = np.int32)
        else:
            self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int32), size=self.lattice.N)
        self._E = self.total_energy()
        self._M = int(self.spins.sum())
    def total_energy(self):
        spins = self.spins
        nn_sum = spins[self.lattice.neighbour].sum(axis=1)    # Forma compatta suggerita da GPT: in pratica NN[i] è una somma di S_j con j nei primi vicini di S_i
        return np.float64(-0.5 * self.J * np.dot(spins, nn_sum))
    def spin_flip(self,idx):
        self.spins[idx] *= -1
    def refresh_observables(self) -> None:
        self._E = self.total_energy()
        self._M = int(self.spins.sum())
    @property
    def energy_density(self):
        return np.float64(self._E / self.lattice.N)
    @property
    def magnetizazion_density(self):
        return np.float64(self._M/ self.lattice.N)
@dataclass
class MetropolisMC:
    """
    Driver Metropolis per il modello di Ising su reticolo triangolare.
    """
    model: Any
    coordination: int = 6
    f90 = jit('mc_metropolis_ising.f90', flags='-O3')  # evita -ffast-math se vuoi riproducibilità

    def __post_init__(self):
        # Setup della simulazione: preparo l'array dei vicini con gli indici fixati, per fortran.
        # Inoltre, preparo già gli array che conterranno l'accettazione del montecarlo
        self._neighbors_1b = np.asfortranarray(self.model.lattice.neighbour + 1, dtype=np.int32)
        self.acc_history: list[int] = []
        self.acc_rate_history: list[float] = []

    @property
    def beta(self) -> float:
        return 1.0 / float(self.model.T)
    
    def sweep(self, beta) -> tuple[int, float]:
        """
        Esegue UN singolo sweep Metropolis e aggiorna E, M del modello.
        Ritorna (acc, acc_rate).
        """
        if beta is None:
            beta = self.beta
        acc, acc_rate = self._fortran_sweep(beta)
        self.acc_history.append(acc)
        self.acc_rate_history.append(acc_rate)
        return acc, acc_rate

    def _fortran_sweep(self, beta: float) -> tuple[int, float]:
        s = self.model.spins  
        J = float(self.model.J)
        acc,acc_rate, dE, dM = f90.metropolis_ising.sweep(
        s, self._neighbors_1b, self.coordination, float(beta), J
        )
        self.model._E += dE
        self.model._M += int(dM)
        return int(acc), float(acc_rate)
    def equilibrate(self, nequil: int) -> None:
        for _ in range(nequil):
            self.sweep(self.beta)

    def sample(self, nmcs: int, measure_every: int = 1):
        """
        Esegue nmcs sweep, misura ogni 'measure_every' sweep.
        Ritorna un dict con serie temporali di m, e, acc_rate.
        """
        N = self.model.lattice.N
        n_samples = (nmcs + measure_every - 1) // measure_every

        m_series = np.empty(n_samples, dtype=np.float64)
        e_series = np.empty(n_samples, dtype=np.float64)
        a_series = np.empty(n_samples, dtype=np.float64)

        t = 0
        for step in range(nmcs):
            acc, acc_rate  = self.sweep(self.beta)
            if (step % measure_every) == 0:
                m_series[t] = self.model._M / N
                e_series[t] = self.model._E / N
                a_series[t] = acc_rate
                t += 1

        # Se nmcs è multiplo perfetto di measure_every, t == n_samples; altrimenti l'ultimo slot è inutilizzato
        if t < n_samples:
            m_series = m_series[:t]
            e_series = e_series[:t]
            a_series = a_series[:t]

        return m_series,e_series,a_series
    def set_temperature(self, T: float) -> None:
        self.model.T = float(T)
    def driver_fixed(self, nequil: int, nmcs: int, measure_step: int):

        nsample = (nmcs + measure_step - 1) // measure_step
        acc, acc_rate, E_array, M_array, acc_sum_out, nsamples_out, spins_array = \
            self.f90.metropolis_ising.many_sweep_measure(
                int(nequil), int(nmcs), int(measure_step), int(nsample),
                self.model.spins, self._neighbors_1b, int(self.coordination),
                float(self.beta), float(self.model.J)
            )

        self.model._M = int(self.model.spins.sum())
        self.model._E = float(self.model.total_energy())   # ricalcolo O(N) ma sempre coerente

        return acc, acc_rate, E_array, M_array, acc_sum_out, nsamples_out, spins_array

    def driver(self, nequil: int, nmcs: int, measure_step: int):
        """
        Esegue equilibrazione + produzione; ritorna serie E/N e M/N campionate,
        insieme all'accettanza totale e al numero effettivo di campioni.
        """

        nsample = (nmcs + measure_step - 1) // measure_step
        acc, acc_rate, E_array, M_array, acc_sum_out, nsamples_out, spins_array = \
            self.f90.metropolis_ising.many_sweep_measure(
                int(nequil), int(nmcs), int(measure_step), int(nsample),
                self.model.spins, self._neighbors_1b, int(self.coordination),
                float(self.beta), float(self.model.J)
            )
        return acc,acc_rate,E_array,M_array,acc_sum_out,nsamples_out,spins_array
    
    def avg_plot(self,array,title:str,average:str):
        """
        Metodo per plottare grandezze scalari durante la simulazione, nei vari nmeasure step
        """
        points = np.arange(len(array))
        fig, ax = plt.subplots(figsize=(8,6),dpi = 150 )
        ax.set_xlabel(f'nsample')
        ax.set_ylabel(average)
        ax.grid(True)
        fig.tight_layout()
        ax.set_title(title)
        ax.legend()
        fig.show()

    