import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_algorithms.optimizers import QNSPSA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
SHOTS = 10  # ← Altere aqui para mudar em todos os lugares

# ==========================================
# 1. INPUT INTERATIVO
# ==========================================
print("Escolha o modo de execução:")
print(f"  [1] Simulação local (FakeVigoV2, shots={SHOTS})")
print(f"  [2] Hardware real da IBM Quantum (shots={SHOTS})")
choice = input("Digite 1 ou 2: ").strip()

# Backend de simulação (sempre usado na otimização)
backend_sim = FakeVigoV2()

# Backend real (só usado na validação final, se escolhido)
if choice == "2":
    USE_REAL_HARDWARE = True
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        token = input("Digite seu IBM Quantum Token: ").strip()
        if not token:
            raise ValueError("Token é obrigatório para hardware real.")
    # Correção: canal correto é "ibm_quantum" (não "ibm_quantum_platform")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend_hw = service.least_busy(operational=True, min_num_qubits=1)
    print(f"Conectado ao hardware real: {backend_hw.name}")
else:
    USE_REAL_HARDWARE = False
    backend_hw = None
    print(f"Usando simulação local (FakeVigoV2, shots={SHOTS})")

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def run_circuit(theta, shots=SHOTS, for_optimization=False):
    """
    Executa o circuito.
    - for_optimization=True: usa sempre FakeVigoV2 (nunca hardware).
    - for_optimization=False: usa hardware real se escolhido, senão FakeVigoV2.
    """
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)

    if for_optimization:
        # Sempre simulação local durante a otimização
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
    counts = result.data.c.get_counts()
    return counts

# Funções para otimização (sempre com simulação)
def fidelity_opt(params, params_pert=None):
    counts = run_circuit(params[0], shots=SHOTS, for_optimization=True)
    total = sum(counts.values())
    return counts.get("0", 0) / total if total > 0 else 0.0

def cost_opt(params):
    return 1.0 - fidelity_opt(params)

# ==========================================
# 3. EXECUÇÃO
# ==========================================
np.random.seed()  # Remove ou fixe (ex: seed(42)) para reprodutibilidade
initial_theta = np.random.uniform(0, 2 * np.pi)
print(f"\n▶ Theta inicial aleatório: {initial_theta:.4f} rad")

# Otimização com FakeVigoV2 (nunca toca no hardware real)
print("Executando otimização com FakeVigoV2 (simulação local)...")
opt = QNSPSA(fidelity=fidelity_opt, maxiter=50, blocking=True, allowed_increase=0.1)
result = opt.minimize(cost_opt, np.array([initial_theta]))
best_theta = result.x[0]
print(f"▶ Theta final (ótimo): {best_theta:.4f} rad")

# Validação final: simulação ou hardware real (apenas 2 execuções)
print("Coletando resultados finais...")
counts_random = run_circuit(initial_theta, shots=SHOTS, for_optimization=False)
counts_opt = run_circuit(best_theta, shots=SHOTS, for_optimization=False)

# ==========================================
# 4. VISUALIZAÇÃO
# ==========================================
states = ["0", "1"]
values_random = [counts_random.get(s, 0) for s in states]
values_opt = [counts_opt.get(s, 0) for s in states]

x = np.arange(len(states))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, values_random, width, label=f"Sem QNSPSA (θ={initial_theta:.2f})", color="#1f77b4", alpha=0.8)
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
print(f"Shots por execução    → {SHOTS}")
print(f"Theta inicial         → {initial_theta:.4f} rad")
print(f"Theta final           → {best_theta:.4f} rad")
print(f"P(0) sem QNSPSA       → {p0_random:.4f}")
print(f"P(0) com QNSPSA       → {p0_opt:.4f}")