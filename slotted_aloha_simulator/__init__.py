"""Top-level package for Slotted Aloha Simulator."""

__author__ = """Fabien Mathieu"""
__email__ = 'loufab@gmail.com'
__version__ = '0.1.0'


from slotted_aloha_simulator.slotted_aloha_simulator import Aloha
from slotted_aloha_simulator.parallel import parallel_compute
from slotted_aloha_simulator.plot import evolution_plot, distribution_plot, live_plot
