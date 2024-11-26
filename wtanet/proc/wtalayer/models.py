"""Process models for WTALayer hierarchical process.

See Also
--------
.process.WTALayer: process which these models implement.

"""

import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.decorator import implements

from lava.proc.wtalayer.process import WTALayer


@implements(proc=WTALayer, protocol=LoihiProtocol)
class SubWTALayerModel(AbstractSubProcessModel):
    def __init__(self, proc:AbstractProcess):
        """Builds WTA Layer structure
        
        See Also
        --------
        .process.WTALayer: process which this model implements.
        """
        shape = proc.proc_params.get("shape", (1, 1))
        inp_weights = proc.proc_params.get("inp_weights", 1)
        exc_weights = proc.proc_params.get("exc_weights", 5)
        inh_weights = proc.proc_params.get("inh_weights", -5)
        du = proc.proc_params.get("du", 4095)
        dv = proc.proc_params.get("dv", 0)
        vth = proc.proc_params.get("vth", 10)
        num_message_bits = proc.proc_params.get("num_message_bits", 1)

        # process weights
        if not (isinstance(inp_weights, np.ndarray) and inp_weights.ndim == 2):
            inp_weights = np.eye(shape[0]) * inp_weights
        if not (isinstance(exc_weights, np.ndarray) and exc_weights.ndim == 2):
            exc_weights = np.eye(shape[0]) * exc_weights
        if not (isinstance(inh_weights, np.ndarray) and inh_weights.ndim == 2):
            inh_weights = (
                np.ones(shape=(shape[0], shape[0])) - np.eye(shape[0])
            ) * inh_weights

        self.dense = Dense(weights=inp_weights, num_message_bits=num_message_bits)
        self.lif = LIF(
            shape=(shape[0],),
            du=du,
            dv=dv,
            vth=vth,
        )
        self.dense_exc = Dense(weights=exc_weights)
        self.dense_inh = Dense(weights=inh_weights)

        proc.in_ports.s_in.connect(self.dense.in_ports.s_in)
        self.dense.out_ports.a_out.connect(self.lif.in_ports.a_in)
        self.lif.out_ports.s_out.connect(proc.out_ports.s_out)
        self.lif.out_ports.s_out.connect(self.dense_exc.in_ports.s_in)
        self.dense_exc.out_ports.a_out.connect(self.lif.in_ports.a_in)
        self.lif.out_ports.s_out.connect(self.dense_inh.in_ports.s_in)
        self.dense_inh.out_ports.a_out.connect(self.lif.in_ports.a_in)

        proc.vars.u.alias(self.lif.vars.u)
        proc.vars.v.alias(self.lif.vars.v)
        proc.vars.du.alias(self.lif.vars.du)
        proc.vars.dv.alias(self.lif.vars.dv)
        proc.vars.vth.alias(self.lif.vars.vth)
        proc.vars.inp_weights.alias(self.dense.vars.weights)
        proc.vars.exc_weights.alias(self.dense_exc.vars.weights)
        proc.vars.inh_weights.alias(self.dense_inh.vars.weights)
        proc.vars.num_message_bits.alias(self.dense.vars.num_message_bits)
