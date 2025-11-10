import numpy as np

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
    
