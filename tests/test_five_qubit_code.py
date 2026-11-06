"""Tests for CNCSim's five-qubit stabilizer code implementation."""

import unittest

import galois
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

from src.five_qubit_code import (
    LOGICAL_X,
    LOGICAL_Z,
    STABILIZERS,
    encode_state,
    encoder_circuit,
    error_correction_circuit,
    error_for_syndrome,
    inject_pauli_error,
    measure_syndrome_with_chp,
    simulate_recovery,
    syndrome_for_error,
    syndrome_extraction_circuit,
)
from src.tableau_helper_functions import (
    pauli_str_to_binary_vec,
    symplectic_inner_product,
)


class FiveQubitCodeTests(unittest.TestCase):
    def test_stabilizers_commute_and_are_independent(self):
        vectors = [pauli_str_to_binary_vec(check) for check in STABILIZERS]
        for row, left in enumerate(vectors):
            for right in vectors[row + 1 :]:
                self.assertEqual(symplectic_inner_product(left, right), 0)
        self.assertEqual(np.linalg.matrix_rank(galois.GF(2)(vectors)), 4)

    def test_logical_operator_relations(self):
        checks = [pauli_str_to_binary_vec(check) for check in STABILIZERS]
        logical_x = pauli_str_to_binary_vec(LOGICAL_X)
        logical_z = pauli_str_to_binary_vec(LOGICAL_Z)
        for check in checks:
            self.assertEqual(symplectic_inner_product(logical_x, check), 0)
            self.assertEqual(symplectic_inner_product(logical_z, check), 0)
        self.assertEqual(symplectic_inner_product(logical_x, logical_z), 1)

    def test_encoder_maps_input_logical_operators_to_code_operators(self):
        circuit = encoder_circuit()
        self.assertEqual(circuit.num_qubits, 5)

        # The synthesized Clifford's stabilizer tableau is the direct check
        # that the logical input and four |0> ancillas map to the code frame.
        from qiskit.quantum_info import Clifford

        encoded = Clifford(circuit)
        actual_stabilizers = encoded.to_dict()["stabilizer"]
        self.assertEqual(actual_stabilizers[0], "+ZZZZZ")
        self.assertEqual(actual_stabilizers[1:], ["+IXZZX", "+XZZXI", "+ZZXIX", "+ZXIXZ"])

    def test_arbitrary_state_encoding_satisfies_checks(self):
        logical_state = (np.sqrt(0.37), np.exp(0.7j) * np.sqrt(0.63))
        encoded = Statevector.from_instruction(encode_state(logical_state))
        for check in STABILIZERS:
            expectation = encoded.expectation_value(Pauli(check[::-1]))
            self.assertAlmostEqual(float(np.real(expectation)), 1.0, places=9)

    def test_all_fifteen_single_qubit_errors_have_unique_syndromes(self):
        errors = [(qubit, pauli) for qubit in range(5) for pauli in "XYZ"]
        syndromes = [syndrome_for_error(qubit, pauli) for qubit, pauli in errors]
        self.assertEqual(len(set(syndromes)), 15)
        for error, syndrome in zip(errors, syndromes):
            self.assertEqual(error_for_syndrome(syndrome), error)

    def test_chp_simulator_measures_all_fifteen_syndromes(self):
        for qubit in range(5):
            for pauli in "XYZ":
                with self.subTest(qubit=qubit, pauli=pauli):
                    self.assertEqual(
                        measure_syndrome_with_chp(qubit, pauli),
                        syndrome_for_error(qubit, pauli),
                    )
        self.assertEqual(measure_syndrome_with_chp(), (0, 0, 0, 0))

    def test_all_fifteen_errors_are_recovered_for_arbitrary_logical_state(self):
        logical_state = (np.sqrt(0.37), np.exp(0.7j) * np.sqrt(0.63))
        for qubit in range(5):
            for pauli in "XYZ":
                with self.subTest(qubit=qubit, pauli=pauli):
                    result = simulate_recovery(logical_state, qubit, pauli)
                    self.assertEqual(result.syndrome, syndrome_for_error(qubit, pauli))
                    self.assertEqual(result.correction, (qubit, pauli))
                    self.assertAlmostEqual(result.fidelity, 1.0, places=9)
                    for expectation in result.stabilizer_expectations:
                        self.assertAlmostEqual(expectation, 1.0, places=9)

    def test_no_error_preserves_basis_and_superposition_inputs(self):
        logical_states = (
            (1.0, 0.0),
            (0.0, 1.0),
            (1 / np.sqrt(2), 1 / np.sqrt(2)),
            (np.sqrt(0.37), np.exp(0.7j) * np.sqrt(0.63)),
        )
        for logical_state in logical_states:
            with self.subTest(logical_state=logical_state):
                result = simulate_recovery(logical_state)
                self.assertEqual(result.syndrome, (0, 0, 0, 0))
                self.assertIsNone(result.correction)
                self.assertAlmostEqual(result.fidelity, 1.0, places=9)

    def test_dynamic_correction_circuit_has_one_branch_per_error(self):
        circuit = error_correction_circuit()
        self.assertEqual(circuit.num_qubits, 9)
        self.assertEqual(circuit.num_clbits, 4)
        self.assertEqual(circuit.count_ops()["if_else"], 15)
        self.assertEqual(circuit.count_ops()["reset"], 4)

    def test_qiskit_dynamic_circuit_corrects_all_fifteen_errors(self):
        backend = AerSimulator()
        correction = error_correction_circuit()
        for qubit in range(5):
            for pauli in "XYZ":
                with self.subTest(qubit=qubit, pauli=pauli):
                    circuit = QuantumCircuit(9, 4)
                    circuit.compose(encoder_circuit(), qubits=range(5), inplace=True)
                    inject_pauli_error(circuit, qubit, pauli)
                    circuit.compose(
                        correction,
                        qubits=range(9),
                        clbits=range(4),
                        inplace=True,
                    )
                    # The correction circuit resets its ancillas, so a second
                    # set of checks directly verifies return to the code space.
                    circuit.compose(
                        syndrome_extraction_circuit(measure=True),
                        qubits=range(9),
                        clbits=range(4),
                        inplace=True,
                    )
                    counts = backend.run(circuit, shots=8).result().get_counts()
                    self.assertEqual(counts, {"0000": 8})

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_state((1.0, 1.0))
        with self.assertRaises(ValueError):
            syndrome_for_error(5, "X")
        with self.assertRaises(ValueError):
            syndrome_for_error(0, "I")
        with self.assertRaises(ValueError):
            error_for_syndrome((0, 1, 0))


if __name__ == "__main__":
    unittest.main()
