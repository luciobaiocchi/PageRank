import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """Represents a single node in a directed graph."""
    
    def __init__(self, number=0, outGoing=None, inGoing=None, link=""):
        """
        Initializes the Node instance.

        Args:
            number (int): A unique identifier for the node.
            outGoing (list, optional): Initial list of outgoing node numbers.
            inGoing (list, optional): Initial list of incoming node numbers.
            link (str, optional): A URL or string associated with the node.
        """
        self.number = number
        self.outGoing = np.array([]) if outGoing is None else np.array(outGoing)
        self.inGoing = np.array([]) if inGoing is None else np.array(inGoing)
        self.link = link
    
    def getOutCount(self):
        """Returns the number of outgoing links (out-degree)."""
        return self.outGoing.size
    
    def getInCount(self):
        """Returns the number of incoming links (in-degree)."""
        return self.inGoing.size
    
    def addOut(self, nodes):
        """
        Adds one or more node numbers to the outgoing links list.

        Args:
            nodes (list or np.array): Node numbers to add.
        """
        nodes = np.array(nodes)
        self.outGoing = np.concatenate((self.outGoing, nodes))
    
    def addIn(self, nodes):
        """
        Adds one or more node numbers to the incoming links list.

        Args:
            nodes (list or np.array): Node numbers to add.
        """
        nodes = np.array(nodes)
        self.inGoing = np.concatenate((self.inGoing, nodes))
        
    def __repr__(self):
        """Returns an unambiguous string representation of the node."""
        return f"Node(number={self.number}, out={self.outGoing}, in={self.inGoing})"
    
class Graph:
    def __init__(self, nodes=None):
        """
        Initializes the Graph instance.

        Args:
            nodes (list, optional): A list of Node objects to initialize the graph with.
        """
        # Set self.nodes to an empty list if nodes is None, otherwise use the provided list
        self.nodes = [] if nodes is None else nodes
    
    def getNode(self, node_number):
        """
        Finds and returns a Node object from the graph by its number.

        Args:
            node_number (int): The 'number' attribute of the Node to find.

        Returns:
            Node: The Node object if found, otherwise None.
        """
        for node in self.nodes:
            if node.number == node_number:
                return node
        return None

    def addNode(self, node):
        """Adds a new Node object to the graph."""
        present = self.getNode(node.number)
        if not present:
            self.nodes.append(node)
    
    def plot(self):
        """
        Plots the graph using networkx and matplotlib.
        
        Note: This requires 'networkx' and 'matplotlib' to be installed.
        (e.g., pip install networkx matplotlib)
        """
        # Create a new directed graph from networkx
        G = nx.DiGraph()

        # Add all nodes to the networkx graph
        # We use node.number as the identifier in networkx
        for node in self.nodes:
            G.add_node(node.number)

        # Add all edges
        for node in self.nodes:
            # node.outGoing contains the numbers of the nodes it points to
            for target_node_number in node.outGoing:
                # Ensure the target node is also in our graph's node list
                # (to avoid adding edges to nodes that don't exist)
                # Cast to int just in case numpy stores it as float
                target_node = self.getNode(int(target_node_number))
                if target_node:
                    # Add an edge from the current node to the target node
                    G.add_edge(node.number, int(target_node_number))

        print("\nPlotting graph... (A new window should open)")
        
        # Draw the graph
        # 'spring_layout' is one of many layout algorithms
        pos = nx.spring_layout(G)
        
        # Draw with options
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                arrowstyle='-|>', arrowsize=20, node_size=700,
                font_weight='bold', arrows=True)
        
        # Display the plot
        plt.show()
        
    def plotLarge(self):
        """
        Plots the graph using networkx and matplotlib, optimized for large datasets.
        """
        G = nx.DiGraph()

        # Aggiungi nodi
        for node in self.nodes:
            G.add_node(node.number, link=node.link) # Conserviamo l'attributo 'link' se necessario

        # Aggiungi archi
        for node in self.nodes:
            for target_node_number in node.outGoing:
                # Conversione sicura a intero
                target_num = int(target_node_number)
                if self.getNode(target_num):
                    G.add_edge(node.number, target_num)

        print(f"\nPlotting graph with {len(G.nodes)} nodes and {len(G.edges)} edges...")
        
        # ----------------------------------------------------
        # --- Ottimizzazioni per Grafi di Grandi Dimensioni ---
        # ----------------------------------------------------
        
        # 1. Calcola il layout (può richiedere tempo per 6000 nodi)
        # La funzione 'fruchterman_reingold_layout' a volte funziona meglio di 'spring_layout'
        pos = nx.spring_layout(G, k=0.1, iterations=50, seed=42)
        
        # 2. Rimuovi le etichette dei nodi (troppo affollato)
        labels = None 
        
        # 3. Riduzione drastica della dimensione dei nodi (da 700 a 5 o meno)
        # Se 5 è ancora troppo, prova 1 o 0.1
        node_size = 5 
        
        # 4. Rendi gli archi sottili e semi-trasparenti
        edge_width = 0.1  # Linea molto sottile
        edge_alpha = 0.4  # Semi-trasparente
        
        # 5. Colore dei nodi (può essere personalizzato in base al grado o al link)
        node_color = 'skyblue'
        
        # Disegna il grafo con le opzioni ottimizzate
        plt.figure(figsize=(12, 12)) # Aumenta la dimensione della finestra di output
        nx.draw(G, pos, 
                with_labels=labels,        # Etichette disattivate
                node_color=node_color,     # Colore nodo
                node_size=node_size,       # Dimensione nodo ridotta
                edge_color='gray',
                width=edge_width,          # Spessore archi ridotto
                alpha=edge_alpha,          # Opacità archi ridotta
                arrowstyle='-',            # Rimuovi frecce per meno confusione
                arrows=False               # Disattiva le frecce se non strettamente necessarie
        )
        
        plt.title(f"Visualizzazione Grafo - {len(G.nodes)} Nodi", fontsize=16)
        plt.axis('off') # Nascondi gli assi
        plt.show()