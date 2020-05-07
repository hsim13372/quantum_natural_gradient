
import numpy as np
import scipy as sp

import qutip as qt
from qutip import Bloch

# State vector we want to plot
psi = np.array([1, 1])
psi = psi/np.linalg.norm(psi)

# Convert to QObject in QuTiP
coords = [qt.Qobj(psi)]

# Settings for Bloch sphere visualization
pt_size = [50]*4
pt_marker = 'o'
pt_color = 'r'

bloch_plot = Bloch()
bloch_plot.point_marker = pt_marker
bloch_plot.point_color = pt_color
bloch_plot.point_size = pt_size

for coords in coords:
    bloch_plot.add_states(coords, 'point')

bloch_plot.show()
