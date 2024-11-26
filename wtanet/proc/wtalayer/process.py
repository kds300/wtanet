"""Winner Take All (WTA) hierarchical process for Lava.

Implementation of a WTA competitive layer between LIF neurons.
Utilizes self-excitatory and inhibitory connections to promote neuron with the
strongest input to fire over other neurons.

"""

from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess


class WTALayer(AbstractProcess):
    """Winner-Take-All layer of LIF neurons.

    Connects a layer of LIF neurons with excitatory and inhibitory connections.
    The default layout has each neuron send excitatory pulses to itself and
    inhibitory pulses to all other neurons upon spiking.

    Attributes
    ----------
    u: numpy.ndarray
        Current values for the LIF neurons
    v: numpy.ndarray
        Voltage values for the LIF neurons
    du: float
        Current decay for the LIF neurons
    dv: float
        Voltage decay for the LIF neurons
    vth: float
        Voltage threshold for the LIF neurons
    inp_weights: float, int, numpy.ndarray
        Weights for the input connections to the neurons.
        Can be a single value, 1-d array, or 2-d array.
        A single value or 1-d array will be converted to a 2-d diagonal array.
    exc_weights: float, int, numpy.ndarray
        Weights for the excitatory connections to the neurons.
        Can be a single value, 1-d array, or 2-d array.
        A single value or 1-d array will be converted to a 2-d diagonal array.
    inh_weights: float, int, numpy.ndarray
        Weights for the inhibitory connectiosn to the neuron.
        Must supply negative values for inhibition.
        Can be a single value, 1-d array, or 2-d array.
        A single value or 1-d array will be converted to a zero-diagonal array.
    num_message_bits: int
        Number of bits for input spikes to the network.
        To use graded input spikes requires num_message_bits > 1.

    See Also
    --------
    lava.proc.lif.process.LIF, lava.proc.dense.process.Dense
    """
    def __init__(self, **kwargs):
        """Create a WTA layer of LIF neurons.

        Parameters
        ----------
        shape : tuple[int]
            Dimensions of the layer. Typically (n, n) for n-neuron WTA layer.
        inp_weights : int, float, array_like, optional
            Connection weights for the input dense process.
            Unless a 2-d array is given, will produce a diagonal array.
        exc_weights: int, float, array_like, optional
            Connection weights for the excitatory dense process.
            Unless a 2-d array is given, will produce a diagonal array.
        inh_weights: int, float, array_like, optional
            Connection weights for the inhibitory dense process. Must supply
            negative values for inhibition.
            Unless a 2-d array is given, will produce a zero-diagonal array.
        du: float, optional
            Current decay parameter for LIF neurons. Applies to entire
            population.
        dv: float, optional
            Voltage decay parameter for LIF neurons. Applies to entire
            population.
        vth: float, optional
            Voltage threshold parameter for LIF neurons. Applies to entire
            population.
        num_message_bits: int, optional
            Number of bits for spikes in the input dense process.
            `num_message_bits == 1` forces binary spikes. For graded spikes,
            set `num_message_bits > 1'

        Example
        -------
        >>> wta = WTALayer(shape=(5, 5), vth=10, exc_weights=5, inh_weights=-5)
        This will create a WTA layer of 5 neurons, with voltage threshold of
        10, self-excitatory weights of 5, and inhibitory weights of -5.
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        inp_weights = kwargs.get("inp_weights", 0)
        exc_weights = kwargs.get("exc_weights", 5)
        inh_weights = kwargs.get("inh_weights", -5)
        du = kwargs.get("du", 4095)
        dv = kwargs.get("dv", 0)
        vth = kwargs.get("vth", 10)
        num_message_bits = kwargs.get("num_message_bits", 1)

        self.s_in = InPort(shape=(shape[1],))
        self.s_out = OutPort(shape=(shape[0],))

        self.u = Var(shape=(shape[0],), init=0)
        self.v = Var(shape=(shape[0],), init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.vth = Var(shape=(1,), init=vth)
        self.inp_weights = Var(shape=shape, init=inp_weights)
        self.exc_weights = Var(shape=shape, init=exc_weights)
        self.inh_weights = Var(shape=shape, init=inh_weights)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)
