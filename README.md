## Welcome to CNCSim Project

#### Introduction
This project is about classical simulation of quantum algorithms with qubits. Quantum computing with magic states (QCM) is a universal model of quantum computing that is an alternative to the usual circuit based model and is driven by stabilizer operations acting on a "magic" input state [[Bravyi-Kitaev, 2004]](https://arxiv.org/abs/quant-ph/0403025).


Here we classically simulate QCM using methods based on sampling from a quasi-probability distribution. These are objects that behave like probability distributions except that their values are allowed to be negative, which can be understood as the onset of "quantumness" into the quantum computation. Howard and Campbell [[Howard-Campbell, 2016]](https://arxiv.org/abs/1609.07488) introduced a quasi-probability representation based on stabilizer states and the cost of classical simulation was controlled by the optimal amount of negativity, called the robustness of magic.

Our simulation approach is based on maximal Closed Non-Contextual (CNC) operators [[Raussendorf et al., 2020]](https://arxiv.org/abs/1905.05374), objects which subsume stabilizer states, therefore promising less negativity and more efficient classical simulation. Central to our implementation is a novel tableau structure, called the *CNC tableau* [[Ipek et al., 2025]](https://arxiv.org/abs/2506.04033). This structure efficiently represents (in the number of qubits, $n$) CNC operators in a tableau of size $\mathcal{O}(n^2)$, much in the same way that the Aaronson-Gottesman tableau [[Aaronson-Gottesman, 2004]](https://arxiv.org/abs/quant-ph/0406196) efficiently represents stabilizer states. The update rules of the CNC tableau under application of Clifford unitaries $\mathcal{O}(n)$ and measurement of Pauli observables $\mathcal{O}(n^2)$ are also efficient.

Maximal CNC operators are also vertices of a class of polytopes introduced in [[Zurel et al., 2020]](https://arxiv.org/abs/2004.01992) for classical simulation of universal quantum computation in the QCM model. An interesting aspect of this approach is that negativity disappears entirely, raising the question of what quantum property is responsible for the presumed hardness of classical simulation. Our study of CNC based simulation in this repository is a first step towards understanding this open problem.

#### Installing Dependencies for User
To install the dependencies for a user, run the following command in the terminal:
```bash
    pip install -r requirements.txt
```


## Patent Pending Notice
This repository contains material related to our arXiv preprint arXiv:2506.04033. Please note that the technology described herein is subject to a pending patent (application number: 2025/007146). Licensed under MIT, which does not grant patent rights. 

For inquiries: cihan.okay@bilkent.edu.tr.

## Five-qubit quantum error correction

CNCSim includes a circuit implementation of the perfect `[[5,1,3]]` stabilizer
code in `src/five_qubit_code.py`. It uses the project Pauli-string convention:
the leftmost character is physical data qubit 0. The stabilizer syndrome is
ordered `(S1, S2, S3, S4)`, where a bit is 1 when the check outcome is -1.

For an exact arbitrary logical input and one Pauli error, use
`simulate_recovery((alpha, beta), qubit, pauli)`. It encodes the logical state,
measures the four checks, decodes the syndrome, applies recovery, then reports
logical-state fidelity and the four post-recovery stabilizer expectations.
`encoder_circuit`, `syndrome_extraction_circuit`, and
`error_correction_circuit` expose the corresponding circuits for composition.
`measure_syndrome_with_chp` executes the stabilizer-input path using the
project's existing `ChpSimulator`.

See `Five_Qubit_Error_Correction.ipynb` for a complete X/Y/Z error example.
The notebook uses Qiskit's exact statevector simulator for arbitrary logical
states and the existing CHP tableau simulator for stabilizer syndromes. The
current `run_qcm` path does not initialize arbitrary logical states or support
general syndrome-controlled recovery conditions, so this workflow does not
route those operations through the CNC QASM runner.

Run its tests with `python -m unittest discover -s tests -v`.
