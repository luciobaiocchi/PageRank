import numpy as np

def PowerMethod(A, N, m=0.15, h=None, relTol=1e-8, maxiter=500):
    """
    Calcola il PageRank usando il Power Method.
    A: Matrice di adiacenza sparsa (column-stochastic, senza dangling links)
    N: Numero di nodi
    m: Probabilità di salto casuale (teleportation, tipicamente 0.15)
    h: Vettore colonna (N,1) binario: 1.0 se il nodo è dangling, 0.0 altrimenti
    """
    # Inizializza s (vettore uniforme)
    s = np.full((N, 1), 1.0 / N)

    # Inizializza xk
    xk = s.copy()
    
    # Se h non è passato, lo calcoliamo (ma meglio passarlo)
    if h is None:
        h = np.zeros((N, 1))

    # Fattore di smorzamento (damping factor)
    alpha = 1.0 - m 

    for k in range(maxiter):
        xk_prev = xk.copy()
        
        # 1. Calcolo massa persa nei nodi dangling
        # h è un vettore di 0 e 1. h.T @ xk somma le probabilità dei nodi che non hanno uscite.
        dangling_sum = h.T @ xk_prev 
        dangling_part = dangling_sum * s # Ridistribuisce la massa persa uniformemente

        # 2. Calcolo nuovo vettore (Google Matrix formula implicta)
        # x_new = alpha * (A * x + massa_dangling) + (1-alpha) * s
        # Nota: (1-m) è alpha.
        xk_new = alpha * (A @ xk_prev + dangling_part) + (m * s)
        
        # 3. Normalizzazione (Opzionale ma consigliata per errori numerici)
        xk_new = xk_new / np.linalg.norm(xk_new, ord=1)
        
        # 4. Check convergenza (Norma L1 della differenza tra vettori)
        # Usiamo la norma del vettore invece dell'autovalore lambda
        diff = np.linalg.norm(xk_new - xk_prev, ord=1)
        
        if diff < relTol:
            print(f"Convergenza raggiunta all'iterazione {k+1} (Diff: {diff:.2e})")
            return xk_new
        
        xk = xk_new

    print(f"Attenzione: Max iterazioni ({maxiter}) raggiunte senza convergenza perfetta.")
    return xk


def testRank(graph):
    A_initial, h, mapping = graph.get_matrix_data()
    N = graph.getCount()

    print(f"Nodi totali: {N}")
    print(f"Nodi dangling: {int(np.sum(h))}")

    # --- Esecuzione PageRank ---
    m = 0.15
    ranks = PowerMethod(A_initial, N, m=m, h=h)
    flat_ranks = ranks.flatten()
    sorted_indices = np.argsort(flat_ranks)[::-1]

    print("\n--- Rank ---")
    print(f"{'Rank':<5} {'Score':<12} {'ID':<6} {'URL'}")
    print("-" * 60)

    for i in range(N):
        idx = sorted_indices[i]
        node_real_id = mapping[idx]
        score = flat_ranks[idx]        
        print(f"{i+1:<5} {score:.8f}   {node_real_id:<6}")