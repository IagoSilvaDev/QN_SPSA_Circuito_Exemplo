import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import QNSPSA

# ==========================================
# Fidelidade — AGORA com 2 parâmetros
# ==========================================
def fidelity(theta, theta_pert=None):
    theta = theta[0]  # sempre vetor, extraímos o parâmetro

    qc = QuantumCircuit(1,1)
    qc.ry(theta, 0)
    qc.measure(0,0)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    counts = sim.run(tqc, shots=2000).result().get_counts()

    return counts.get("0", 0) / 2000  # queremos o estado |0>


# ==========================================
# Custo baseado em fidelidade
# ==========================================
def cost(params):
    return 1 - fidelity(params)


# ==========================================
# Configurando QNSPSA
# ==========================================
opt = QNSPSA(
    fidelity=fidelity,   # ← agora no formato correto
    maxiter=60,
    blocking=True,
    allowed_increase=0.1
)

initial = np.array([4.0])

result = opt.minimize(cost, initial)

print("\n===== RESULTADO FINAL =====")
print("Theta final    →", result.x[0])
print("Custo final    →", cost(result.x))
print("Fidelidade     →", fidelity(result.x))
