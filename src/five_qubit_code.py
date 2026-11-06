"""Circuit and stabilizer tools for the perfect five-qubit code.

Pauli strings in this module are written in physical-qubit order: the leftmost
character acts on data qubit 0. This matches ``tableau_helper_functions``.
Qiskit Pauli labels are reversed at the boundary because Qiskit labels place
qubit 0 at the right.

The code produces ordinary Qiskit Clifford circuits. ``ChpSimulator`` is used
to execute syndrome extraction for stabilizer inputs, while ``Statevector``
provides exact checks for arbitrary one-qubit logical inputs. The current QCM
QASM runner has a more restricted input and feed-forward model, so it is not
used as the execution backend here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Clifford, Pauli, Statevector, partial_trace, state_fidelity
from qiskit.synthesis import synth_clifford_full

from src.chp import ChpSimulator
from src.tableau_helper_functions import (
    pauli_str_to_binary_vec,
    symplectic_inner_product,
)


NUM_DATA_QUBITS = 5
NUM_STABILIZERS = 4
NUM_SYNDROME_QUBITS = 4

STABILIZERS = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
LOGICAL_X = "XXXXX"
LOGICAL_Z = "ZZZZZ"

# Dual Pauli generators D_i satisfy [D_i, S_j] = delta_ij and commute with
# both logical operators. They define the Clifford encoder together with S_i.
_DESTABILIZERS = ("ZIZII", "ZZZZI", "ZZIZZ", "IZIIZ")
_ANCILLA_OFFSET = NUM_DATA_QUBITS


@dataclass(frozen=True)
class RecoveryResult:
    """Exact result for one encoded-state, error, syndrome, and recovery run."""

    syndrome: Tuple[int, int, int, int]
    correction: Optional[Tuple[int, str]]
    fidelity: float
    stabilizer_expectations: Tuple[float, float, float, float]


def _validate_pauli(pauli: str) -> str:
    pauli = pauli.upper()
    if pauli not in {"X", "Y", "Z"}:
        raise ValueError("pauli must be one of 'X', 'Y', or 'Z'.")
    return pauli


def _validate_qubit(qubit: int, num_qubits: int = NUM_DATA_QUBITS) -> int:
    if not isinstance(qubit, (int, np.integer)) or not 0 <= int(qubit) < num_qubits:
        raise ValueError(f"qubit must be an integer in [0, {num_qubits - 1}].")
    return int(qubit)


def _qiskit_pauli_label(pauli_string: str) -> str:
    """Convert this module's q0-leftmost strings to Qiskit's label order."""
    return pauli_string[::-1]


@lru_cache(maxsize=1)
def _synthesized_encoder() -> QuantumCircuit:
    """Synthesize a Clifford that maps |0> ancillas and logical qubit 0
    onto the five-qubit code space.
    """
    # Qiskit's tableau lists the images of X_i first, then Z_i, in qubit-index
    # order. Logical input is q0; input qubits q1..q4 are encoding ancillas.
    destabilizers = (LOGICAL_X,) + _DESTABILIZERS
    stabilizers = (LOGICAL_Z,) + STABILIZERS
    tableau = {
        "destabilizer": ["+" + _qiskit_pauli_label(row) for row in destabilizers],
        "stabilizer": ["+" + _qiskit_pauli_label(row) for row in stabilizers],
    }
    return synth_clifford_full(Clifford.from_dict(tableau))


def encoder_circuit() -> QuantumCircuit:
    """Return a five-qubit encoder.

    Put the logical input on qubit 0 and initialize qubits 1 through 4 to
    |0>. The returned Clifford circuit encodes the input into the code space.
    """
    return _synthesized_encoder().copy()


def encode_state(logical_state: Iterable[complex]) -> QuantumCircuit:
    """Build a circuit encoding an arbitrary normalized one-qubit state.

    ``logical_state`` is ``(alpha, beta)`` for alpha|0> + beta|1>. The circuit
    prepares that state on qubit 0, then applies the five-qubit Clifford
    encoder; qubits 1 through 4 begin in |0>.
    """
    amplitudes = np.asarray(tuple(logical_state), dtype=complex)
    if amplitudes.shape != (2,):
        raise ValueError("logical_state must contain exactly (alpha, beta).")
    if not np.all(np.isfinite(amplitudes)) or not np.isclose(
        np.linalg.norm(amplitudes), 1.0, atol=1e-10
    ):
        raise ValueError("logical_state amplitudes must be finite and normalized.")

    circuit = QuantumCircuit(NUM_DATA_QUBITS, name="five_qubit_encode")
    circuit.append(StatePreparation(amplitudes), [0])
    circuit.compose(encoder_circuit(), qubits=range(NUM_DATA_QUBITS), inplace=True)
    return circuit


def syndrome_for_error(qubit: int, pauli: str) -> Tuple[int, int, int, int]:
    """Return the binary syndrome, with 1 denoting a -1 check eigenvalue.

    Syndrome bits are ordered S1, S2, S3, S4. A bit is 1 exactly when the
    injected Pauli anticommutes with that stabilizer.
    """
    qubit = _validate_qubit(qubit)
    pauli = _validate_pauli(pauli)
    error = ["I"] * NUM_DATA_QUBITS
    error[qubit] = pauli
    error_vector = pauli_str_to_binary_vec("".join(error))
    return tuple(
        int(symplectic_inner_product(error_vector, pauli_str_to_binary_vec(check)))
        for check in STABILIZERS
    )


def error_for_syndrome(
    syndrome: Iterable[int],
) -> Optional[Tuple[int, str]]:
    """Decode a four-bit syndrome to its single-qubit Pauli correction.

    The all-zero syndrome returns ``None``. Invalid or ambiguous syndromes
    raise ``ValueError`` rather than silently selecting a recovery.
    """
    bits = tuple(syndrome)
    if len(bits) != NUM_STABILIZERS or any(bit not in (0, 1, False, True) for bit in bits):
        raise ValueError("syndrome must contain four binary values in S1..S4 order.")
    bits = tuple(int(bit) for bit in bits)
    if bits == (0, 0, 0, 0):
        return None

    matches = [
        (qubit, pauli)
        for qubit in range(NUM_DATA_QUBITS)
        for pauli in ("X", "Y", "Z")
        if syndrome_for_error(qubit, pauli) == bits
    ]
    if len(matches) != 1:
        raise ValueError(f"syndrome {bits} does not identify one single-qubit error.")
    return matches[0]


def inject_pauli_error(circuit: QuantumCircuit, qubit: int, pauli: str) -> None:
    """Append an X, Y, or Z error to a physical data qubit in-place."""
    qubit = _validate_qubit(qubit)
    pauli = _validate_pauli(pauli)
    getattr(circuit, pauli.lower())(qubit)


def syndrome_extraction_circuit(measure: bool = True) -> QuantumCircuit:
    """Return Clifford circuitry that extracts all four stabilizer checks.

    Data qubits are 0..4 and syndrome ancillas are 5..8. A measured version
    has a four-bit classical register whose bit i is the outcome for S(i+1).
    With ``measure=False`` the ancillas hold the syndrome in computational
    basis, allowing exact Statevector simulation.
    """
    circuit = QuantumCircuit(
        NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS,
        NUM_SYNDROME_QUBITS if measure else 0,
        name="five_qubit_syndrome",
    )

    for check_index, check in enumerate(STABILIZERS):
        ancilla = _ANCILLA_OFFSET + check_index
        for qubit, pauli in enumerate(check):
            if pauli == "I":
                continue
            if pauli == "X":
                circuit.h(qubit)
            elif pauli == "Y":
                # H S† maps a Y eigenbasis state to the Z eigenbasis.
                circuit.sdg(qubit)
                circuit.h(qubit)

            circuit.cx(qubit, ancilla)

            if pauli == "X":
                circuit.h(qubit)
            elif pauli == "Y":
                circuit.h(qubit)
                circuit.s(qubit)

        if measure:
            circuit.measure(ancilla, check_index)

    return circuit


def recovery_circuit(syndrome: Iterable[int]) -> QuantumCircuit:
    """Return a five-qubit circuit applying the correction for ``syndrome``."""
    circuit = QuantumCircuit(NUM_DATA_QUBITS, name="five_qubit_recovery")
    correction = error_for_syndrome(syndrome)
    if correction is not None:
        inject_pauli_error(circuit, *correction)
    return circuit


def error_correction_circuit() -> QuantumCircuit:
    """Return an ancilla-based, syndrome-controlled correction circuit.

    Compose this nine-qubit/four-classical-bit circuit after encoding and any
    data error. It measures S1..S4 and conditionally applies the matching
    single-qubit Pauli correction using Qiskit's dynamic-circuit conditions.
    """
    quantum_register = QuantumRegister(
        NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS, "q"
    )
    syndrome_register = ClassicalRegister(NUM_SYNDROME_QUBITS, "syndrome")
    circuit = QuantumCircuit(
        quantum_register,
        syndrome_register,
        name="five_qubit_error_correction",
    )
    measured_checks = syndrome_extraction_circuit(measure=True)
    circuit.compose(
        measured_checks,
        qubits=range(NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS),
        clbits=list(syndrome_register),
        inplace=True,
    )

    for syndrome in range(1, 1 << NUM_STABILIZERS):
        bits = tuple((syndrome >> i) & 1 for i in range(NUM_STABILIZERS))
        correction = error_for_syndrome(bits)
        if correction is None:
            continue
        qubit, pauli = correction
        with circuit.if_test((syndrome_register, syndrome)):
            inject_pauli_error(circuit, qubit, pauli)
    for ancilla in range(_ANCILLA_OFFSET, _ANCILLA_OFFSET + NUM_SYNDROME_QUBITS):
        circuit.reset(ancilla)
    return circuit


def measure_syndrome_with_chp(
    qubit: Optional[int] = None,
    pauli: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    """Run syndrome extraction for encoded |0_L> using CNCSim's CHP engine.

    This is a stabilizer-state execution path: the logical input is |0>,
    while arbitrary logical states are checked by ``simulate_recovery``.
    """
    if (qubit is None) != (pauli is None):
        raise ValueError("provide both qubit and pauli, or neither for no error.")

    num_qubits = NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(encoder_circuit(), qubits=range(NUM_DATA_QUBITS), inplace=True)
    if qubit is not None and pauli is not None:
        inject_pauli_error(circuit, qubit, pauli)
    circuit.compose(syndrome_extraction_circuit(measure=False), inplace=True)

    # CHP supports H, S, and CNOT. Transpilation decomposes Clifford gates
    # generated by the encoder (e.g. swaps and Pauli gates) into that basis.
    clifford_circuit = transpile(circuit, basis_gates=["h", "s", "cx"], optimization_level=0)
    simulator = ChpSimulator(num_qubits)
    for instruction in clifford_circuit.data:
        gate = instruction.operation.name
        qubits = [clifford_circuit.find_bit(bit).index for bit in instruction.qubits]
        if gate == "h":
            simulator.hadamard(qubits[0])
        elif gate == "s":
            simulator.phase(qubits[0])
        elif gate == "cx":
            simulator.cnot(qubits[0], qubits[1])
        elif gate != "id":
            raise ValueError(f"CHP syndrome path cannot execute gate {gate!r}.")

    outcomes = tuple(
        int(simulator.measure(_ANCILLA_OFFSET + i, bias=0.0).value)
        for i in range(NUM_SYNDROME_QUBITS)
    )
    return outcomes


def simulate_recovery(
    logical_state: Iterable[complex],
    qubit: Optional[int] = None,
    pauli: Optional[str] = None,
) -> RecoveryResult:
    """Exactly simulate encoding, one optional error, syndrome, and recovery.

    ``logical_state`` is ``(alpha, beta)``. ``qubit`` and ``pauli`` specify
    the optional physical error. The returned fidelity compares the recovered
    data register with the ideal encoded logical state, ignoring global phase.
    """
    if (qubit is None) != (pauli is None):
        raise ValueError("provide both qubit and pauli, or neither for no error.")
    if qubit is not None and pauli is not None:
        qubit = _validate_qubit(qubit)
        pauli = _validate_pauli(pauli)

    ideal_circuit = encode_state(logical_state)
    ideal_state = Statevector.from_instruction(ideal_circuit)

    workflow = QuantumCircuit(NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS)
    workflow.compose(ideal_circuit, qubits=range(NUM_DATA_QUBITS), inplace=True)
    if qubit is not None and pauli is not None:
        inject_pauli_error(workflow, qubit, pauli)
    workflow.compose(syndrome_extraction_circuit(measure=False), inplace=True)
    measured_state = Statevector.from_instruction(workflow)

    ancilla_qubits = list(range(_ANCILLA_OFFSET, _ANCILLA_OFFSET + NUM_SYNDROME_QUBITS))
    syndrome_probabilities = measured_state.probabilities(qargs=ancilla_qubits)
    syndrome_index = int(np.argmax(syndrome_probabilities))
    if not np.isclose(syndrome_probabilities[syndrome_index], 1.0, atol=1e-9):
        raise RuntimeError("Stabilizer measurements did not produce a deterministic syndrome.")
    syndrome = tuple(
        (syndrome_index >> i) & 1 for i in range(NUM_SYNDROME_QUBITS)
    )

    correction = error_for_syndrome(syndrome)
    if correction is not None:
        inject_pauli_error(workflow, *correction)
    recovered_state = measured_state.evolve(recovery_circuit_on_nine_qubits(syndrome))
    data_state = partial_trace(recovered_state, ancilla_qubits)
    fidelity = float(state_fidelity(ideal_state, data_state))
    expectations = tuple(
        float(np.real(data_state.expectation_value(Pauli(_qiskit_pauli_label(check)))))
        for check in STABILIZERS
    )

    return RecoveryResult(
        syndrome=syndrome,
        correction=correction,
        fidelity=fidelity,
        stabilizer_expectations=expectations,
    )


def recovery_circuit_on_nine_qubits(syndrome: Iterable[int]) -> QuantumCircuit:
    """Return a recovery circuit on data plus four idle syndrome ancillas."""
    circuit = QuantumCircuit(NUM_DATA_QUBITS + NUM_SYNDROME_QUBITS)
    correction = error_for_syndrome(syndrome)
    if correction is not None:
        inject_pauli_error(circuit, *correction)
    return circuit
