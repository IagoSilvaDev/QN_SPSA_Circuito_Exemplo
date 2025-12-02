import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_algorithms.optimizers import QNSPSA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
SHOTS = 300  # ← Altere aqui para mudar em todos os lugares

# ==========================================
# 1. INPUT INTERATIVO
# ==========================================
print("Escolha o modo de execução:")
print(f"  [1] Simulação local (FakeVigoV2, shots={SHOTS})")
print(f"  [2] Hardware real da IBM Quantum (shots={SHOTS})")
choice = input("Digite 1 ou 2: ").strip()

if choice == "2":
    USE_REAL_HARDWARE = True
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        token = input("Digite seu IBM Quantum Token: ").strip()
        if not token:
            raise ValueError("Token é obrigatório para hardware real.")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.least_busy(operational=True, min_num_qubits=1)
    print(f"✅ Conectado ao hardware real: {backend.name}")
else:
    USE_REAL_HARDWARE = False
    backend = FakeVigoV2()
    print(f"✅ Usando simulação local (FakeVigoV2, shots={SHOTS})")

sampler = SamplerV2(backend)

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def run_circuit(theta, shots=SHOTS):
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    if USE_REAL_HARDWARE:
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=1)
        tqc = pm.run(qc)
    else:
        from qiskit import transpile
        tqc = transpile(qc, backend, optimization_level=0)
    
    job = sampler.run([tqc], shots=shots)
    result = job.result()[0]
    counts = result.data.c.get_counts()
    return counts

def fidelity(params, params_pert=None):
    counts = run_circuit(params[0], shots=SHOTS)
    total = sum(counts.values())
    return counts.get("0", 0) / total if total > 0 else 0.0

def cost(params):
    return 1.0 - fidelity(params, None)

# ==========================================
# 3. EXECUÇÃO
# ==========================================
initial_theta = 3.0
print(f"\n▶ Theta inicial: {initial_theta:.2f}")

counts_random = run_circuit(initial_theta, shots=SHOTS)
opt = QNSPSA(fidelity=fidelity, maxiter=50, blocking=True, allowed_increase=0.1)
result = opt.minimize(cost, np.array([initial_theta]))
best_theta = result.x[0]
counts_opt = run_circuit(best_theta, shots=SHOTS)

# ==========================================
# 4. VISUALIZAÇÃO
# ==========================================
states = ["0", "1"]
values_random = [counts_random.get(s, 0) for s in states]
values_opt = [counts_opt.get(s, 0) for s in states]

x = np.arange(len(states))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, values_random, width, label=f"Sem QNSPSA (θ={initial_theta})", color="#1f77b4", alpha=0.8)
plt.bar(x + width/2, values_opt,    width, label=f"Com QNSPSA (θ={best_theta:.2f})", color="#ff7f0e", alpha=0.8)

plt.xticks(x, states)
plt.xlabel("Resultado da medição")
plt.ylabel(f"Contagens (shots={SHOTS})")
plt.title("Efeito do QNSPSA: maximizando P(|0⟩) com RY(θ)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 5. RESULTADOS
# ==========================================
total_random = sum(counts_random.values())
total_opt = sum(counts_opt.values())
p0_random = counts_random.get("0", 0) / total_random if total_random > 0 else 0.0
p0_opt = counts_opt.get("0", 0) / total_opt if total_opt > 0 else 0.0

print("\n===== RESULTADOS =====")
print(f"Shots utilizados      → {SHOTS}")
print(f"Theta inicial         → {initial_theta:.2f}")
print(f"Theta final           → {best_theta:.2f}")
print(f"P(0) sem QNSPSA       → {p0_random:.4f}")
print(f"P(0) com QNSPSA       → {p0_opt:.4f}")