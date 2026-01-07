import pandas as pd
import numpy as np
from functions.graph import Graph, Node
from functions.functions import PowerMethod, testRank


print("================================================================ FIG 2.1 CASE ================================================================")
# Create the nodes
node1 = Node(number=1)
node2 = Node(number=2)
node3 = Node(number=3)
node4 = Node(number=4)

# linking edges
node1.addOut([2,3,4]) 
node1.addIn([3,4]) 
node2.addIn([1]) 
node2.addOut([3,4]) 
node3.addIn([1,2,4]) 
node3.addOut([1]) 
node4.addIn([2,1]) 
node4.addOut([1,3]) 

graph = Graph([node1, node2, node3, node4])
graph.print()
#graph.plot()
testRank(graph)
    
print("================================================================ FIG 2.2 CASE ================================================================")
node5 = Node(number=5)
node3.addOut([5])
node3.addIn([5])
node5.addOut([3])
node5.addIn([3])
graph.addNode(node5)

testRank(graph)
graph.print()
#graph.plot()
print("================================================================== HOLLINS ===================================================================")
try:
    f = open("/Users/luciobaiocchi/polito/0_Algebra/PageRank/hollins.dat")
    first_line = f.readline().split(" ")
    url_count = int(first_line[0])
    content = f.readlines()
    urls = content[0:url_count] # Urls salvati per dopo
    connections = content[url_count:]
    f.close()
except FileNotFoundError:
    print("Errore: File non trovato. Controlla il percorso.")
    exit()

source = []
dest = []
for connection in connections:
    line = connection.split(" ")
    if len(line) >= 2:
        source.append(int(line[0]))
        dest.append(int(line[1]))

df = pd.DataFrame({"source": source, "destination": dest})

# --- Costruzione Grafo ---
df_outgoing = df.groupby('source')['destination'].apply(lambda x: x.to_numpy())
all_ids = np.unique(np.concatenate((df['source'].unique(), df['destination'].unique())))

graph = Graph()

for node_id in all_ids:
    node = Node(number=node_id)
    if node_id in df_outgoing.index:
        node.addOut(df_outgoing.loc[node_id])
    graph.addNode(node)

# --- Preparazione Matrici ---
print("Generazione matrice sparsa...")
A_initial, h, mapping = graph.get_matrix_data()
N = graph.getCount()

print(f"Nodi totali: {N}")
print(f"Nodi dangling: {int(np.sum(h))}")

# --- Esecuzione PageRank ---
m = 0.15
ranks = PowerMethod(A_initial, N, m=m, h=h)

# --- Risultati ---
flat_ranks = ranks.flatten()
sorted_indices = np.argsort(flat_ranks)[::-1]

print("\n--- Top 10 PageRank ---")
print(f"{'Rank':<5} {'Score':<12} {'ID':<6} {'URL'}")
print("-" * 60)

for i in range(10):
    idx = sorted_indices[i]
    node_real_id = mapping[idx]
    score = flat_ranks[idx]
    
    # Recupera URL in formato stringa
    if node_real_id <= len(urls):
        url_text = urls[node_real_id - 1].strip()
    else:
        url_text = "URL not found"
        
    print(f"{i+1:<5} {score:.8f}   {node_real_id:<6} {url_text}")
    
# OPTIONAL DECOMMENT THE FOLLOWING LINE IN ORDER
# TO SEE IMAGE OF GRAPH, REQUIRES plotly INSTALLED
#graph.plotLargePlotly()