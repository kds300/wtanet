# wtanet
Winner-Take-All Network implemented in Lava

The `WTALayer` is a hierarchical process model for [Lava](https://github.com/lava-nc/lava).
It consists of a layer of Leaky Integrate and Fire (LIF) neurons recurrently connected through excitatory and inhibitory synapses.
The resulting layer creates competition between the neurons, promoting the neuron with the largest input while suppressing the other neurons.

## Setup
First, set up Lava in your environment.
Once this is done, the easiest way to use the `WTALayer` is to copy the dir `/wtanet/proc/wtalayer/` into the Lava process dir (`lava/proc/`).
The process can then be imported like any other Lava process:

```python
from lava.proc.wtalayer.process import WTALayer

wta = WTALayer(shape=(5,5), vth=10, exc_weights=5, inh_weights=-5)
```
