import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_algorithms.optimizers import QNSPSA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os

SHOTS = 300  
N_QUBITS = 2
NUM_PARAMS = 2

print("Escolha o modo de execução:")
print(f"  [1] Simulação local (FakeVigoV2, shots={SHOTS})")
print(f"  [2] Hardware real da IBM Quantum (shots={SHOTS})")
choice = input("Digite 1 ou 2: ").strip()

backend_sim = FakeVigoV2()

if choice == "2":
    USE_REAL_HARDWARE = True
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        token = input("Digite seu IBM Quantum Token: ").strip()
        if not token:
            raise ValueError("Token é obrigatório para hardware real.")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend_hw = service.least_busy(operational=True, min_num_qubits=N_QUBITS)
    print(f"Conectado ao hardware real: {backend_hw.name}")
else:
    USE_REAL_HARDWARE = False
    backend_hw = None
    print(f"Usando simulação local (FakeVigoV2, shots={SHOTS})")

def create_bell_ansatz(params):
    qc = QuantumCircuit(2, 2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def run_circuit_generic(qc, shots=SHOTS, for_optimization=False):
    """Executa circuito no backend apropriado."""
    if for_optimization:
        tqc = transpile(qc, backend_sim, optimization_level=0)
        sampler = SamplerV2(backend_sim)
    else:
        if USE_REAL_HARDWARE:
            pm = generate_preset_pass_manager(target=backend_hw.target, optimization_level=1)
            tqc = pm.run(qc)
            sampler = SamplerV2(backend_hw)
        else:
            tqc = transpile(qc, backend_sim, optimization_level=0)
            sampler = SamplerV2(backend_sim)
    
    job = sampler.run([tqc], shots=shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    return counts

def fidelity_opt(params, params_pert=None):
    qc = create_bell_ansatz(params)
    counts = run_circuit_generic(qc, shots=SHOTS, for_optimization=True)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p00 = counts.get("00", 0) / total
    p11 = counts.get("11", 0) / total
    balance = 1.0 - abs(p00 - p11)
    success = p00 + p11
    return max(0.0, success * balance)


def cost_opt(params):
    return 1.0 - fidelity_opt(params)

initial_params = np.random.uniform(0, np.pi, NUM_PARAMS) 
print(f"\n▶ Iniciando otimização com {NUM_PARAMS} parâmetros (2 qubits)...")

opt = QNSPSA(fidelity=fidelity_opt, maxiter=100, blocking=True, allowed_increase=0.1)
result = opt.minimize(cost_opt, initial_params)
best_params = result.x
best_fidelity = fidelity_opt(best_params)
print(f"▶ Fidelidade final (simulação): {best_fidelity:.4f}")

print("Coletando resultados finais...")
qc_initial = create_bell_ansatz(initial_params)
qc_best = create_bell_ansatz(best_params)

counts_random = run_circuit_generic(qc_initial, shots=SHOTS, for_optimization=False)
counts_opt = run_circuit_generic(qc_best, shots=SHOTS, for_optimization=False)

all_states = ["00", "01", "10", "11"]
values_random = [counts_random.get(s, 0) for s in all_states]
values_opt = [counts_opt.get(s, 0) for s in all_states]

x = np.arange(len(all_states))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, values_random, width, label="Antes da otimização", color="#1f77b4", alpha=0.8)
plt.bar(x + width/2, values_opt,    width, label="Após QNSPSA", color="#ff7f0e", alpha=0.8)

plt.xticks(x, all_states)
plt.xlabel("Resultado da medição (2 qubits)")
plt.ylabel(f"Contagens (shots={SHOTS})")
plt.title("Otimização para o estado de Bell |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.close()

def bell_fidelity(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p00 = counts.get("00", 0)
    p11 = counts.get("11", 0)
    return (p00 + p11) / total

f_random = bell_fidelity(counts_random)
f_opt = bell_fidelity(counts_opt)

print("\n===== RESULTADOS =====")
print(f"Shots por execução    → {SHOTS}")
print(f"Parâmetros otimizados → {NUM_PARAMS}")
print(f"Fidelidade |Φ⁺⟩ (antes)  → {f_random:.4f}")
print(f"Fidelidade |Φ⁺⟩ (depois) → {f_opt:.4f}")
print(f"Melhoria              → {f_opt - f_random:.4f}")