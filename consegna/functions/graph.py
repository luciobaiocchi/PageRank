import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
import time
from scipy.sparse import csr_matrix
import plotly.graph_objects as go

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
    
    def getVector(self):
        """
        Return the vector used to calculate the rank of page k, using the relative formula, where:
        - x_j is a page that link to page x_k(the one we are calculating the rank vector)

        Returns:
        Return the vector composed by the rank of each incoming node, devided by the outgoing link from that node:
        ex:
            n_o = number_outgoning_nodes
            node = 1; in = node3, node4 ; out = node3, node4, node2;
            rank_1 = n_o(node3) + n_o(node4)
            vector = [0, 0, 1/n_o(3), 1/n_o(4)]
        """
        #for in_node in self.inGoing:
            
        
        return 0
class Graph:
    def __init__(self, nodes=None):
        """
        Initializes the Graph instance.

        Args:
            nodes (list, optional): A list of Node objects to initialize the graph with.
        """
        # Set self.nodes to an empty list if nodes is None, otherwise use the provided list
        self.nodes = [] if nodes is None else nodes
        # Mappa per convertire ID nodo (es. 1, 28, 100) in indici di matrice (0, 1, 2...)
        self.id_to_index = {} 
        self.index_to_id = {}
    
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
            
    def getCount(self):
        """Returnd the count of nodes"""
        return len(self.nodes)
    
    def getList(self):
        """Returns a list of all node numbers."""
        # List comprehension: more concise and faster
        return [n.number for n in self.nodes]
    
    def get_matrix_data(self):
        """Genera matrice sparsa A e vettore dangling h"""
        nodes = self.nodes
        N = len(nodes)
        self.id_to_index = {node.number: i for i, node in enumerate(nodes)}
        self.index_to_id = {i: node.number for i, node in enumerate(nodes)}

        rows, cols, data = [], [], []
        h = np.zeros((N, 1))

        for i, node in enumerate(nodes):
            out_neighbors = node.outGoing 
            out_degree = len(out_neighbors)

            if out_degree == 0:
                h[i, 0] = 1.0 # Nodo dangling
            else:
                weight = 1.0 / out_degree
                for target_id in out_neighbors:
                    if target_id in self.id_to_index:
                        target_idx = self.id_to_index[target_id]
                        rows.append(target_idx)
                        cols.append(i)
                        data.append(weight)

        A = csr_matrix((data, (rows, cols)), shape=(N, N))
        return A, h, self.index_to_id

    
    def print(self):
        for node in self.nodes:
            print(node)
            
    def _calculate_arrow_geometry(self, start_pos, end_pos, node_radius, shift_offset=0):
        """
        Calcola le coordinate esatte di inizio e fine freccia.
        - Accorcia la linea per non finire sotto il nodo (node_radius).
        - Sposta la linea lateralmente se serve (shift_offset).
        """
        vector = end_pos - start_pos
        dist = np.linalg.norm(vector)
        if dist == 0: return start_pos, end_pos
        unit_vector = vector / dist
        perp_vector = np.array([-unit_vector[1], unit_vector[0]])
        start_coord = start_pos + (unit_vector * node_radius) + (perp_vector * shift_offset)
        end_coord = end_pos - (unit_vector * node_radius) + (perp_vector * shift_offset)

        return start_coord, end_coord

    def plot(self):
        print("\nGenerazione grafico stabile con layout circolare...")
        
        n_nodes = self.getCount()
        if n_nodes == 0: return

        G_nx = nx.DiGraph()
        for node in self.nodes:
            G_nx.add_node(node.number)
            for target in node.outGoing:
                G_nx.add_edge(node.number, int(target))

        pos = nx.circular_layout(G_nx)
        NODE_SIZE = 45
        NODE_RADIUS_DATA = 0.12 
        SHIFT_AMOUNT = 0.05

        node_x, node_y, node_text, hover_text = [], [], [], []
        
        for node_num, coords in pos.items():
            node_x.append(coords[0])
            node_y.append(coords[1])
            node_text.append(str(node_num))
            
            node_obj = self.getNode(node_num)
            info = f"<b>Nodo {node_num}</b><br>Out: {node_obj.getOutCount()}<br>In: {node_obj.getInCount()}"
            hover_text.append(info)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text, textposition="middle center",
            hoverinfo='text', hovertext=hover_text,
            # Marker grande e opaco per nascondere l'inizio delle frecce se i calcoli non sono perfetti
            marker=dict(size=NODE_SIZE, color='#7bc0f7', line=dict(width=2, color='#333')), 
            textfont=dict(size=14, color='black', weight='bold')
        )

        annotations = []
        processed_pairs = set()

        for source_node in self.nodes:
            u = source_node.number
            if u not in pos: continue
            p_u = np.array(pos[u])

            for target_num in source_node.outGoing:
                v = int(target_num)
                if v not in pos: continue
                if u == v: continue # Ignora self-loop per ora

                pair_key = tuple(sorted((u, v)))
                
                # Se abbiamo già disegnato questa coppia, saltiamo
                if pair_key in processed_pairs: continue

                p_v = np.array(pos[v])
                target_obj = self.getNode(v)
                
                # Check Bidirezionale
                is_bidirectional = u in target_obj.outGoing

                if not is_bidirectional:
                    s_coord, e_coord = self._calculate_arrow_geometry(p_u, p_v, NODE_RADIUS_DATA, 0)
                    
                    annotations.append(dict(
                        ax=s_coord[0], ay=s_coord[1], axref='x', ayref='y',
                        x=e_coord[0], y=e_coord[1], xref='x', yref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='black'
                    ))
                else:
                    s_uv, e_uv = self._calculate_arrow_geometry(p_u, p_v, NODE_RADIUS_DATA, SHIFT_AMOUNT)
                    annotations.append(dict(
                        ax=s_uv[0], ay=s_uv[1], axref='x', ayref='y',
                        x=e_uv[0], y=e_uv[1], xref='x', yref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='black'
                    ))
                    s_vu, e_vu = self._calculate_arrow_geometry(p_v, p_u, NODE_RADIUS_DATA, SHIFT_AMOUNT)
                    annotations.append(dict(
                        ax=s_vu[0], ay=s_vu[1], axref='x', ayref='y',
                        x=e_vu[0], y=e_vu[1], xref='x', yref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='black'
                    ))
                    
                    processed_pairs.add(pair_key)
        axis_range = [-1.3, 1.3]

        fig = go.Figure(data=[node_trace],
            layout=go.Layout(
                title=dict(text='Visualizzazione grafo', font=dict(size=16)),
                showlegend=False, hovermode='closest',
                margin=dict(b=20,l=20,r=20,t=50),
                annotations=annotations,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=axis_range, scaleanchor="y", scaleratio=1),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=axis_range),
                plot_bgcolor='white', width=700, height=700
            ))
        fig.show()
        
    def plotLarge(self):
        """
        Plots the graph using networkx and matplotlib, optimized for large datasets.
        Displays two separate plots: one for incoming edges and one for outgoing edges.
        The plots are interactive and zoomable with high resolution.
        """
        G = nx.DiGraph()

        # Aggiungi nodi
        for node in self.nodes:
            G.add_node(node.number, link=node.link)

        # Aggiungi archi
        for node in self.nodes:
            for target_node_number in node.outGoing:
                target_num = int(target_node_number)
                if self.getNode(target_num):
                    G.add_edge(node.number, target_num)

        print(f"\nPlotting graph with {len(G.nodes)} nodes and {len(G.edges)} edges...")
        
        # Calcola il layout una sola volta per mantenere la stessa disposizione in entrambi i grafici
        pos = nx.spring_layout(G, k=0.1, iterations=50, seed=42)
        
        # Calcola in-degree e out-degree per ogni nodo
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Crea una figura con alta risoluzione (DPI aumentato)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 24), dpi=250)
        
        # ============================================
        # GRAFICO 1: Nodi colorati per IN-DEGREE
        # ============================================
        
        # Colora i nodi in base al numero di collegamenti in entrata
        # Usa una scala logaritmica per migliorare il contrasto
        max_in = max(in_degrees.values()) if in_degrees else 1
        node_colors_in = []
        for node in G.nodes():
            degree = in_degrees.get(node, 0)
            # Normalizzazione con scala logaritmica per aumentare il contrasto
            if degree == 0:
                normalized = 0
            else:
                normalized = np.log1p(degree) / np.log1p(max_in)
            # Scala da 0.3 a 1.0 per evitare colori troppo chiari
            node_colors_in.append(0.3 + normalized * 0.7)
        
        nx.draw(G, pos, 
                ax=ax1,
                with_labels=False,
                node_color=node_colors_in,
                cmap=plt.cm.Reds,
                vmin=0.0,
                vmax=1.0,
                node_size=30,
                edge_color='#333333',
                width=0.3,
                alpha=0.7,
                arrowstyle='->',
                arrowsize=8,
                arrows=True
        )
        
        ax1.set_title(f"Nodi colorati per Collegamenti IN ENTRATA (Blu intenso = più collegamenti)\n{len(G.nodes)} Nodi, {len(G.edges)} Archi", 
                    fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Aggiungi colorbar per il primo grafico
        sm1 = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_in))
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Numero di collegamenti in entrata', rotation=270, labelpad=20, fontsize=12)
        
        # ============================================
        # GRAFICO 2: Nodi colorati per OUT-DEGREE
        # ============================================
        
        # Colora i nodi in base al numero di collegamenti in uscita
        max_out = max(out_degrees.values()) if out_degrees else 1
        node_colors_out = []
        for node in G.nodes():
            degree = out_degrees.get(node, 0)
            # Normalizzazione con scala logaritmica per aumentare il contrasto
            if degree == 0:
                normalized = 0
            else:
                normalized = np.log1p(degree) / np.log1p(max_out)
            # Scala da 0.3 a 1.0 per evitare colori troppo chiari
            node_colors_out.append(0.3 + normalized * 0.7)
        
        nx.draw(G, pos, 
                ax=ax2,
                with_labels=False,
                node_color=node_colors_out,
                cmap=plt.cm.Blues,
                vmin=0.0,
                vmax=1.0,
                node_size=30,
                edge_color='#333333',
                width=0.3,
                alpha=0.7,
                arrowstyle='->',
                arrowsize=8,
                arrows=True
        )
        
        ax2.set_title(f"Nodi colorati per Collegamenti IN USCITA (blu intenso = più collegamenti)\n{len(G.nodes)} Nodi, {len(G.edges)} Archi", 
                    fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')
        
        # Aggiungi colorbar per il secondo grafico
        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_out))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Numero di collegamenti in uscita', rotation=270, labelpad=20, fontsize=12)
        
        # Abilita la modalità interattiva per lo zoom
        plt.tight_layout()
        print("\n" + "="*60)
        print("USO INTERATTIVO:")
        print("="*60)
        print("🔍 Usa il pulsante 'ZOOM' nella toolbar per ingrandire")
        print("👆 Usa il pulsante 'PAN' per muoverti nel grafico")
        print("🏠 Clicca 'HOME' per tornare alla vista originale")
        print("💾 Clicca 'SAVE' per salvare l'immagine ad alta risoluzione")
        print("="*60)
        
        plt.show()
        
    def plotLargePlotly(self):
        """
        Plots the graph using Plotly for interactive visualization.
        You can drag nodes, zoom, and interact with the graph.
        Displays two separate plots: one for incoming edges and one for outgoing edges.
        Uses parallel processing for faster graph creation.
        """

        
        start_time = time.time()
        
        G = nx.DiGraph()

        print(f"Building graph structure...")
        # Aggiungi nodi
        for node in self.nodes:
            G.add_node(node.number, link=node.link)

        # Aggiungi archi
        for node in self.nodes:
            for target_node_number in node.outGoing:
                target_num = int(target_node_number)
                if self.getNode(target_num):
                    G.add_edge(node.number, target_num)

        print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
        print(f"Computing layout using parallel processing with {cpu_count()} CPUs...")
        
        # Calcola il layout con più iterazioni per grafi grandi
        # Usa n_jobs=-1 se disponibile (solo in alcune versioni)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        layout_time = time.time()
        print(f"Layout computed in {layout_time - start_time:.2f} seconds")
        
        # Calcola in-degree e out-degree per ogni nodo
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Crea subplot con 2 grafici
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Collegamenti IN ENTRATA (rosso = più collegamenti)', 
                        'Collegamenti IN USCITA (blu = più collegamenti)'),
            vertical_spacing=0.1
        )
        
        # ============================================
        # FUNZIONE HELPER PER CREARE GLI ARCHI
        # ============================================
        def create_edge_trace(G, pos):
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            return edge_trace
        
        print("Creating visualization traces...")
        
        # ============================================
        # GRAFICO 1: IN-DEGREE
        # ============================================
        
        # Archi per grafico 1
        edge_trace_1 = create_edge_trace(G, pos)
        fig.add_trace(edge_trace_1, row=1, col=1)
        
        # Nodi per grafico 1 (colorati per in-degree)
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        
        max_in = max(in_degrees.values()) if in_degrees else 1
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            in_deg = in_degrees.get(node, 0)
            # Scala logaritmica per migliore visualizzazione
            if in_deg == 0:
                color_val = 0
            else:
                color_val = np.log1p(in_deg) / np.log1p(max_in)
            node_colors.append(color_val)
            
            node_text.append(f'Nodo: {node}<br>In: {in_deg}<br>Out: {out_degrees.get(node, 0)}')
        
        node_trace_1 = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="In-Degree",
                    xanchor="left",
                    x=1.02
                ),
                line=dict(width=0.5, color='white')
            ),
            showlegend=False
        )
        
        fig.add_trace(node_trace_1, row=1, col=1)
        
        # ============================================
        # GRAFICO 2: OUT-DEGREE
        # ============================================
        
        # Archi per grafico 2
        edge_trace_2 = create_edge_trace(G, pos)
        fig.add_trace(edge_trace_2, row=2, col=1)
        
        # Nodi per grafico 2 (colorati per out-degree)
        node_colors_out = []
        max_out = max(out_degrees.values()) if out_degrees else 1
        
        for node in G.nodes():
            out_deg = out_degrees.get(node, 0)
            # Scala logaritmica per migliore visualizzazione
            if out_deg == 0:
                color_val = 0
            else:
                color_val = np.log1p(out_deg) / np.log1p(max_out)
            node_colors_out.append(color_val)
        
        node_trace_2 = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors_out,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="Out-Degree",
                    xanchor="left",
                    x=1.02
                ),
                line=dict(width=0.5, color='white')
            ),
            showlegend=False
        )
        
        fig.add_trace(node_trace_2, row=2, col=1)
        
        # ============================================
        # CONFIGURAZIONE LAYOUT
        # ============================================
        
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
        fig.update_layout(
            title=dict(
                text=f"Visualizzazione Interattiva Grafo - {len(G.nodes)} Nodi, {len(G.edges)} Archi",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=1400,
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        total_time = time.time() - start_time
        print(f"Visualization created in {total_time:.2f} seconds total")
        
        print("\n" + "="*70)
        print("CONTROLLI INTERATTIVI:")
        print("="*70)
        print("🖱️  TRASCINA i nodi per spostarli")
        print("🔍 SCROLL per zoomare")
        print("👆 CLICK E TRASCINA per muovere l'intera vista")
        print("🏠 DOPPIO CLICK per resettare la vista")
        print("💾 Usa i bottoni in alto a destra per salvare o altre opzioni")
        print("="*70)
        
        fig.show()