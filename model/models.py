import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import logging
from time import sleep
from abc import ABCMeta

from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.plugin.qiskit.qiskit_plugin import tq2qiskit

# from quark import Task
from model.utils import INIT_METHOD, orthogonal_init, gen_task, calculate_all_Z_expectations


logger = logging.getLogger(__name__)


class DiscreteActor(nn.Module): 
    def __init__(self, is_quantum:bool,*args, **kwargs) -> None:
        super().__init__()
        self.is_quantum = is_quantum
        if self.is_quantum == True:
            self._init_quantum(*args, **kwargs)
        elif self.is_quantum ==False:
            self._init_classical(*args, **kwargs)
        else: raise ValueError(f"is_quantum should be a bool, but got {self.is_quantum}")

    def _init_quantum(self, n_wires: int, n_blocks: int, input_dim: int, output_dim: int, ini_method, is_critic=False,
                 use_quafu=False, use_quafu_simulator=True, quantum_device='Dongling') -> None:
        if ini_method[0] != "NOT":
            self.pre_encoding_net = nn.Linear(input_dim, n_wires)
            INIT_METHOD[ini_method[0]](self.pre_encoding_net)
        else:
            if n_wires != input_dim:
                n_wires = input_dim
            self.pre_encoding_net = nn.Identity()

        self.pqc_layer = PQCLayer(n_wires=n_wires, n_blocks=n_blocks, use_quafu=use_quafu, use_quafu_simulator=use_quafu_simulator, quantum_device=quantum_device)
        self.post_processing_net = nn.Linear(n_wires, output_dim)
        if 'O' == ini_method[-1]:
            INIT_METHOD[ini_method[-1]](self.post_processing_net, 0.01)
        else:
            INIT_METHOD[ini_method[-1]](self.post_processing_net)

        self.is_critic = is_critic
        self.softmax_layer = nn.Softmax(dim=1)
    def _init_classical(self, input_dim: int,hidden_dims: [int], output_dim: int,use_orthogonal_init: bool = True, is_critic=False) ->None:
        layers = []
        sizes = [input_dim] + hidden_dims
        layers.append(nn.Flatten())
        for i  in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i],sizes[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-1],output_dim))
        self.net = nn.Sequential(*layers)

        if use_orthogonal_init:
            gain = 1.0
            for layer in reversed(self.net):
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=gain)
                    gain = np.sqrt(2)

        self.is_critic = is_critic
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        if self.is_quantum == True:
            x = self.pre_encoding_net(x)
            x = self.pqc_layer(x)
            x = self.post_processing_net(x)
        else: 
            x = self.net(x)
        if not self.is_critic:
            x = self.softmax_layer(x)
        return x


class Critic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        if len(kwargs) > 4:
            # Initialize quantum model if more than 4 arguments are passed
            self._init_quantum_model(**kwargs)
            self.quantum = True
        else:
            # Initialize classical model otherwise
            self._init_classical_model(**kwargs)
            self.quantum = False

    def _init_classical_model(self, state_dim: int, hidden_dims: [int], activation: nn.Module = None,
                              use_orthogonal_init: bool = True) -> None:
        if activation is None:
            activation = nn.ReLU

        self.net = nn.Sequential()
        for in_features, out_features in zip([state_dim, *hidden_dims], hidden_dims + [1]):
            self.net.append(nn.Linear(in_features, out_features, bias=True))
            if out_features != 1:
                self.net.append(activation())

        if use_orthogonal_init:
            gain = 1.0
            for layer in reversed(self.net):
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer, gain=gain)
                    gain = np.sqrt(2)

        self.is_critic = True

    def _init_quantum_model(self, n_wires: int, n_blocks: int, input_dim: int, output_dim: int, ini_method,
                            is_critic=True, use_quafu=False, use_quafu_simulator=True,
                            quantum_device='Dongling') -> None:
        if ini_method[0] != "NOT":
            self.pre_encoding_net = nn.Linear(input_dim, n_wires)
            INIT_METHOD[ini_method[0]](self.pre_encoding_net)
        else:
            if n_wires != input_dim:
                n_wires = input_dim
            self.pre_encoding_net = nn.Identity()

        self.pqc_layer = PQCLayer(
            n_wires=n_wires,
            n_blocks=n_blocks,
            use_quafu=use_quafu,
            use_quafu_simulator=use_quafu_simulator,
            quantum_device=quantum_device
        )
        self.post_processing_net = nn.Linear(n_wires, output_dim)
        if 'O' == ini_method[-1]:
            INIT_METHOD[ini_method[-1]](self.post_processing_net, 0.01)
        else:
            INIT_METHOD[ini_method[-1]](self.post_processing_net)

        self.is_critic = is_critic
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        if self.quantum:
            x = self.pre_encoding_net(x)
            x = self.pqc_layer(x)
            x = self.post_processing_net(x)
        else:x = self.net(x) 
        if not self.is_critic:
            x = self.softmax_layer(x)
        return x


class ContinuousActor(nn.Module):
    def __init__(self, is_quantum:bool,*args, **kwargs) -> None:
        super().__init__()
        self.is_quantum = is_quantum
        if self.is_quantum == True:
            self._init_quantum(*args, **kwargs)
        elif self.is_quantum == False:
            self._init_classical(*args, **kwargs)
        else: raise ValueError(f"is_quantum should be a bool, but got {self.is_quantum}")

    def _init_quantum(self, n_wires: int, n_blocks: int, input_dim: int, output_dim: int, ini_method) -> None:

        if ini_method[0] != "NOT":
            self.pre_encoding_net = nn.Linear(input_dim, n_wires)
            INIT_METHOD[ini_method[0]](self.pre_encoding_net)
        else:
            if n_wires != input_dim:
                n_wires = input_dim
            self.pre_encoding_net = nn.Identity()

        self.pqc_layer = PQCLayer(n_wires=n_wires, n_blocks=n_blocks)
        self.alpha_layer = nn.Linear(n_wires, output_dim)
        self.beta_layer = nn.Linear(n_wires, output_dim)
        if 'O' == ini_method[-1]:
            INIT_METHOD[ini_method[-1]](self.alpha_layer, 0.01)
            INIT_METHOD[ini_method[-1]](self.beta_layer, 0.01)
        else:
            INIT_METHOD[ini_method[-1]](self.alpha_layer)
            INIT_METHOD[ini_method[-1]](self.beta_layer)

    def _init_classical(self, input_dim: int,hidden_dims: [int], output_dim: int,use_orthogonal_init: bool = True, is_critic=False) ->None:
        layers = []
        sizes = [input_dim] + hidden_dims
        layers.append(nn.Flatten())
        for i  in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i],sizes[i+1]))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.alpha_layer = nn.Linear(sizes[-1], output_dim)
        self.beta_layer = nn.Linear(sizes[-1], output_dim)
        gain = 1.0
        if use_orthogonal_init:
            orthogonal_init(self.alpha_layer, gain=gain)
            orthogonal_init(self.beta_layer, gain=gain)
            for layer in reversed(self.net):
                if isinstance(layer, nn.Linear):
                    gain = np.sqrt(2)
                    orthogonal_init(layer, gain=gain)

        self.is_critic = is_critic


    def forward(self, s):
        if self.is_quantum == True:
            s = self.pre_encoding_net(s)
            s = self.pqc_layer(s)
            # alpha and beta need to be larger than 1, so we use 'softplus' as the activation function and then plus 1
            alpha = F.softplus(self.alpha_layer(s)) + 1.0
            beta = F.softplus(self.beta_layer(s)) + 1.0
        else: 
            s = self.net(s)
            alpha = F.softplus(self.alpha_layer(s)) + 1.0
            beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist


class OP12All(tq.QuantumModule):
    def __init__(self, n_wires: int, has_params=False, trainable=False, init_params=None, op_fun=None):
        super().__init__()
        self.n_wires = n_wires
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op_fun(has_params=has_params, trainable=trainable, init_params=init_params))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            self.ops_all[k](q_device, wires=k)


class RX2All(OP12All):
    def __init__(self, n_wires: int, has_params=False, trainable=False, init_params=None):
        super().__init__(n_wires, has_params, trainable, init_params, tq.RX)


class RY2All(OP12All):
    def __init__(self, n_wires: int, has_params=False, trainable=False, init_params=None):
        super().__init__(n_wires, has_params, trainable, init_params, tq.RY)


class RZ2All(OP12All):
    def __init__(self, n_wires: int, has_params=False, trainable=False, init_params=None):
        super().__init__(n_wires, has_params, trainable, init_params, tq.RZ)


class H2All(OP12All):
    def __init__(self, n_wires: int, has_params=False, trainable=False, init_params=None):
        super().__init__(n_wires, has_params, trainable, init_params, tq.Hadamard)


class CZ2All(tq.QuantumModule):
    def __init__(self, n_wires: int, has_params=False, trainable=False, circular=False, init_params=None):
        super().__init__()
        self.n_wires = n_wires
        self.circular = circular
        self.ops_all = tq.QuantumModuleList()
        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - 1
        for k in range(n_ops):
            self.ops_all.append(tq.CZ(has_params=has_params, trainable=trainable, init_params=init_params))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, (k + 1) % self.n_wires]
            self.ops_all[k](q_device, wires=wires)


class CN2All(tq.QuantumModule):
    def __init__(self, n_wires: int, has_params=False, trainable=False, circular=False, init_params=None):
        super().__init__()
        self.n_wires = n_wires
        self.circular = circular
        self.ops_all = tq.QuantumModuleList()
        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - 1
        for k in range(n_ops):
            self.ops_all.append(tq.CNOT(has_params=has_params, trainable=trainable, init_params=init_params))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, (k + 1) % self.n_wires]
            self.ops_all[k](q_device, wires=wires)


class RzRyVariationalLayer(tq.QuantumModule):
    def __init__(self, n_wires: int, with_entangle: bool = True) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.with_entangle = with_entangle
        self.rz_layer = RZ2All(n_wires=self.n_wires, has_params=True, trainable=True, init_params=torch.pi * torch.rand(1))
        self.ry_layer = RY2All(n_wires=self.n_wires, has_params=True, trainable=True, init_params=torch.pi * torch.rand(1))
        if n_wires == 2:
            circular = False
        else:
            circular = True
        self.cz_layer = CZ2All(n_wires=self.n_wires, circular=circular)

    def forward(self, q_device: tq.QuantumDevice) -> None:
        self.rz_layer(q_device)
        self.ry_layer(q_device)
        if self.with_entangle:
            self.cz_layer(q_device)


class RxRyRzVariationalLayer(tq.QuantumModule):
    def __init__(self, n_wires: int, with_entangle: bool = True) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.with_entangle = with_entangle
        self.rx_layer = RX2All(n_wires=self.n_wires, has_params=True, trainable=True, init_params=torch.pi * torch.rand(1))
        self.ry_layer = RY2All(n_wires=self.n_wires, has_params=True, trainable=True, init_params=torch.pi * torch.rand(1))
        self.rz_layer = RZ2All(n_wires=self.n_wires, has_params=True, trainable=True, init_params=torch.pi * torch.rand(1))
        if n_wires == 2:
            circular = False
        else:
            circular = True
        self.cz_layer = CZ2All(n_wires=self.n_wires, circular=circular)

    def forward(self, q_device: tq.QuantumDevice) -> None:
        self.rx_layer(q_device)
        self.ry_layer(q_device)
        self.rz_layer(q_device)
        if self.with_entangle:
            self.cz_layer(q_device)


class PQCLayer(tq.QuantumModule):
    def __init__(self, n_wires: int, n_blocks: int, use_quafu=False, use_quafu_simulator=True, quantum_device='Dongling') -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_blocks = n_blocks
        self.H_layer = H2All(n_wires)
        self.vqc_blocks = tq.QuantumModuleList()
        self.dqc_blocks = tq.QuantumModuleList()
        self.use_quafu = use_quafu
        self.use_quafu_simulator = use_quafu_simulator
        for _ in range(self.n_blocks):
            self.vqc_blocks.append(RzRyVariationalLayer(n_wires=self.n_wires))
            self.dqc_blocks.append(RyRzScaleEncoder(n_wires=self.n_wires))
        self.vqc_blocks.append(RzRyVariationalLayer(n_wires=self.n_wires, with_entangle=False))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.quantum_device = quantum_device
        self.ALL_COUNT = 0

    def _forward(self, x):
        bsz = x.shape[0]
        q_dev = tq.QuantumDevice(n_wires=self.n_wires, device=x.device, bsz=bsz)
        self.H_layer(q_dev)
        for vqc_block, dqc_block in zip(self.vqc_blocks, self.dqc_blocks):
            vqc_block(q_dev)
            dqc_block(q_dev, x)
        self.vqc_blocks[-1](q_dev)
        x = self.measure(q_dev)
        return x

    def forward(self, x):
        if self.use_quafu:
            return self._forward_quafu(x)
        else:
            return self._forward(x)

    def to_qiskit_circuit(self, x):
        bsz = x.shape[0]
        q_dev = tq.QuantumDevice(n_wires=self.n_wires, device=x.device, bsz=bsz)
        circs = []
        dqc_blocks_circ_list = [dqc_block.to_qiskit(x) for dqc_block in self.dqc_blocks]
        for k in range(bsz):
            circ = QuantumCircuit(self.n_wires)
            q_dev.reset_states(bsz=bsz)
            circ = circ.compose(tq2qiskit(q_dev, self.H_layer))
            for vqc_block, dqc_block in zip(self.vqc_blocks, dqc_blocks_circ_list):
                circ = circ.compose(tq2qiskit(q_dev, vqc_block))
                circ = circ.compose(dqc_block[k])
            circ = circ.compose(tq2qiskit(q_dev, self.vqc_blocks[-1]))
            circs.append(circ)
        return circs

    # def _forward_quafu(self, x):
    #     try:
    #         import yaml
    #         with open('config.yaml', 'r') as f:
    #             config = yaml.safe_load(f)
    #         if 'key' in config:
    #             user_key = config['key']
    #         else:
    #             raise ValueError("The 'key' field is missing in the 'config.yaml' configuration file.")
    #     except FileNotFoundError:
    #         raise FileNotFoundError("The configuration file 'config.yaml' does not exist.")

    #     tmgr = Task(user_key)
    #     circs = self.to_qiskit_circuit(x)
    #     qasm_str_list = [dumps(circuit) for circuit in circs]
    #     task_list = [gen_task(qasm_str, device=self.quantum_device) for qasm_str in qasm_str_list]

    #     logger.info(f"{self.quantum_device} is calculating {len(task_list)} tasks")
    #     self.ALL_COUNT += len(task_list)
    #     logger.info(f"Total calculation of {self.ALL_COUNT} tasks")

    #     with open('./task_id.txt', 'w') as f:
    #         f.write(f'qasm_str_list: {qasm_str_list}\n')
    #         f.write(f'tid_list: {task_list}\n')

    #     res_list = []
    #     for idx, task in enumerate(task_list):
    #         logger.info(f'Task {idx + 1} starts calculating')
    #         tid = tmgr.run(task, repeat=1)
    #         res = {}
    #         while res.get('status', 0) != 'Finished':
    #             res = tmgr.result(tid)
    #             if not isinstance(res, dict):
    #                 res = {}
    #             logger.info(f'Task {idx + 1} is still calculating')
    #             sleep(3)
    #         res_list.append(res)
    #         logger.info(f'Task {idx + 1} calculation completed')

    #     res = np.array([calculate_all_Z_expectations(res['count']) for res in res_list])
    #     res = torch.tensor(res, dtype=torch.float).to(x.device)
    #     have_grad = self._forward(x)
    #     delta = have_grad.detach() - res
    #     assert not delta.requires_grad, "The difference participates in gradient propagation!"
    #     return have_grad + delta


class ScaleEncoder(tq.Encoder, metaclass=ABCMeta):
    def __init__(self, n_wires: int, route='rx') -> None:
        super().__init__()
        self.route = route
        self.n_wires = n_wires
        self.n_lamda = torch.nn.Parameter(torch.ones(n_wires))
        self.func_list = self._gen_func_list()

    @tq.static_support
    def forward(self, q_dev: tq.QuantumDevice, x) -> None:
        x = self._get_scaled_input(x)
        for info in self.func_list:
            if tq.op_name_dict[info["func"]].num_params > 0:
                params = x[:, info["input_idx"]]
            else:
                params = None

            func_name_dict[info["func"]](
                q_dev,
                wires=info["wires"],
                params=params,
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def _gen_func_list(self):
        func_list = [
            {'input_idx': [i], 'func': self.route, 'wires': [i]}
            for i in range(self.n_wires)
        ]
        return func_list

    def _get_scaled_input(self, x):
        return torch.tanh(torch.einsum('j, ij -> ij', self.n_lamda, x))

    def to_qiskit(self, x):
        # assuming the x is in batch mode
        x = self._get_scaled_input(x)
        bsz = x.shape[0]

        circ_list = []
        for k in range(bsz):
            circ = QuantumCircuit(self.n_wires)
            for info in self.func_list:
                if info["func"] == "rx":
                    circ.rx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ry":
                    circ.ry(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rz":
                    circ.rz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rxx":
                    circ.rxx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ryy":
                    circ.ryy(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzz":
                    circ.rzz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzx":
                    circ.rzx(x[k][info["input_idx"][0]].item(), *info["wires"])
                else:
                    raise NotImplementedError(info["func"])
            circ_list.append(circ)

        return circ_list


class RyRzScaleEncoder(tq.Encoder):
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder1 = ScaleEncoder(n_wires, 'ry')
        self.encoder2 = ScaleEncoder(n_wires, 'rz')

    def forward(self, q_dev: tq.QuantumDevice, x) -> None:
        self.encoder1(q_dev, x)
        self.encoder2(q_dev, x)

    def to_qiskit(self, x):
        circ_list1 = self.encoder1.to_qiskit(x)
        circ_list2 = self.encoder2.to_qiskit(x)

        return [c1.compose(c2) for c1, c2 in zip(circ_list1, circ_list2)]

