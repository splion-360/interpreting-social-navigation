import numpy as np
from helper import getVector
from multiprocessing import Pool

class ST_GRAPH():

    def __init__(self, batch_size = 50, seq_length = 5, nkeys = 12):
        '''
        Initializer function for the ST graph class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        '''
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.nkeys = nkeys
    
        self.nodes = [{} for _ in range(batch_size)]
        self.edges = [{} for _ in range(batch_size)]

    def reset(self):
        self.nodes = [{} for _ in range(self.batch_size)]
        self.edges = [{} for _ in range(self.batch_size)]

    def readGraph(self,  source_batch, num_workers = 4): 
        self.source_batch = source_batch
        pool = Pool(processes = num_workers)
        workers = []
        
        for sequence in range(self.batch_size): 
            workers.append(pool.apply_async(self.process, args = (sequence, )))
        
        pool.close()
        pool.join()

        for result in workers: 
            nodes, edges, sequence = result.get()
            self.nodes[sequence] = nodes
            self.edges[sequence] = edges

    def process(self, sequence):
        '''
        Main function that constructs the ST graph from the batch data
        params:
        source_batch : List of lists of numpy arrays. Each numpy array corresponds to a frame in the sequence.
        '''
        nodes, edges = {}, {}
        source_seq = self.source_batch[sequence]
        
        for framenum in range(self.seq_length):
            # Each frame is a numpy array
            # each row in the array is of the form
            # miceID, keypoints, x, y
            frame = source_seq[framenum]

            # Add nodes
            for mice in range(frame.shape[0]):
                for keypoint in range(frame.shape[1]): 
                    miceID, kpID = mice, keypoint
                    x = frame[mice,keypoint,1]
                    y = frame[mice,keypoint,2]
                    pos = (x, y)
                    node_id = int((miceID * self.nkeys) + keypoint)

                    if node_id not in nodes:
                        node_type = 'M'
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        nodes[node_id] = ST_NODE(node_type, node_id, node_pos_list)
                    else:
                        nodes[node_id].addPosition(pos, framenum)

                        # Add Temporal edge between the node at current time-step
                        # and the node at previous time-step
                        edge_id = (int(miceID * self.nkeys + kpID), int(miceID * self.nkeys + kpID))
                        pos_edge = (nodes[node_id].getPosition(framenum-1), pos)
                        if edge_id not in edges:
                            edge_type = 'M-M/T'
                            edge_pos_list = {}
                            # ASSUMPTION: Adding temporal edge at the later time-step
                            edge_pos_list[framenum] = pos_edge
                            edges[edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                        else:
                            edges[edge_id].addPosition(pos_edge, framenum)

            for ped_in in range(frame.shape[0]):
                for kp_in in range(frame.shape[1]): 
                    for ped_out in range(ped_in + 1, frame.shape[0]): 
                        for kp_out in range(frame.shape[1]): 

                            pedID_in = frame[ped_in, kp_in, 0]
                            pedID_out = frame[ped_out, kp_out, 0]
                            pos_in = (frame[ped_in, kp_in, 1], frame[ped_in, kp_out, 2])
                            pos_out = (frame[ped_out, kp_out, 1], frame[ped_out, kp_out, 2])
                            pos = (pos_in, pos_out)
                            edge_id = (int(pedID_in * self.nkeys + kp_in), int(pedID_out * self.nkeys + kp_out))

                            
                            if edge_id not in edges:
                                edge_type = 'M-M/S'
                                edge_pos_list = {}
                                edge_pos_list[framenum] = pos
                                edges[edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                            else:
            
                                edges[edge_id].addPosition(pos, framenum)
        return nodes, edges, sequence
        
    def printGraph(self):
        '''
        Print function for the graph
        For debugging purposes
        '''
        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print('Printing Nodes')
            print('===============================')
            for node in nodes.values():
                node.printNode()
                print('--------------')

            print
            print('Printing Edges')
            print('===============================')
            for edge in edges.values():
                edge.printEdge()
                print('--------------')

    def getSequence(self):
        '''
        Gets the sequence
        '''
        nodes = self.nodes[0]
        edges = self.edges[0]
        
        numNodes = len(nodes.keys())
        numEdges =  self.nkeys * ((2 * self.nkeys) + 1) ## will always be a constant 
        ##(Right now, we are not considering the edges within mice to avoid compute constraints)
        ## Total edges: 3 X 288 + 3 X 12 = 3 X 300 


        retNodes = np.zeros((self.seq_length, numNodes, 2))
        retEdges = np.zeros((self.seq_length, 3 * numEdges, 2))  # Diagonal contains temporal edges

        retNodePresent = [[] for _ in range(self.seq_length)]
        retEdgePresent = [[] for _ in range(self.seq_length)]

        for i, mice in enumerate(nodes.keys()):
            pos_list = nodes[mice].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append(i)
                    retNodes[framenum, mice, :] = list(pos_list[framenum])
           
        for mice_i, mice_j in edges.keys():
            edge = edges[(mice_i, mice_j)]
           
            if mice_i == mice_j:
                # Temporal edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((mice_i, mice_j))
                        retEdges[framenum, mice_i * (2 * self.nkeys) + mice_j, :] = getVector(edge.edge_pos_list[framenum])
                                   
            else:
                # Spatial edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((mice_i, mice_j))
                        retEdgePresent[framenum].append((mice_j, mice_i))
                   
                        retEdges[framenum, mice_i * (2 * self.nkeys) + mice_j, :] = getVector(edge.edge_pos_list[framenum])
                        retEdges[framenum, mice_j * (2 * self.nkeys) + mice_i, :] = -np.copy(retEdges[framenum, mice_i * (2 * self.nkeys) + mice_j, :])
                          
        return retNodes, retEdges, retNodePresent, retEdgePresent

    def printGraph(self):
        '''
        Print function for the graph
        For debugging purposes
        '''
        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print('Printing Nodes')
            print('===============================')
            for node in nodes.values():
                node.printNode()
                print('--------------')

            print
            print('Printing Edges')
            print('===============================')
            for edge in edges.values():
                edge.printEdge()
                print('--------------')

    def getSequence(self):
        '''
        Gets the sequence
        '''
        nodes = self.nodes[0]
        edges = self.edges[0]
        
        numNodes = len(nodes.keys())
        numEdges =  self.nkeypoints * ((2 * self.nkeypoints) + 1) ## will always be a constant 
        ##(Right now, we are not considering the edges within mice to avoid compute constraints)
        ## Total edges: 3 X 288 + 3 X 12 = 3 X 300 


        retNodes = np.zeros((self.seq_length, numNodes, 2))
        retEdges = np.zeros((self.seq_length, 3 * numEdges, 2))  # Diagonal contains temporal edges

        retNodePresent = [[] for _ in range(self.seq_length)]
        retEdgePresent = [[] for _ in range(self.seq_length)]

        for i, mice in enumerate(nodes.keys()):
            pos_list = nodes[mice].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append(i)
                    retNodes[framenum, mice, :] = list(pos_list[framenum])
           
        for mice_i, mice_j in edges.keys():
            edge = edges[(mice_i, mice_j)]
           
            if mice_i == mice_j:
                # Temporal edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((mice_i, mice_j))
                        retEdges[framenum, mice_i * (2 * self.nkeypoints) + mice_j, :] = getVector(edge.edge_pos_list[framenum])
                                   
            else:
                # Spatial edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((mice_i, mice_j))
                        retEdgePresent[framenum].append((mice_j, mice_i))
                   
                        retEdges[framenum, mice_i * (2 * self.nkeypoints) + mice_j, :] = getVector(edge.edge_pos_list[framenum])
                        retEdges[framenum, mice_j * (2 * self.nkeypoints) + mice_i, :] = -np.copy(retEdges[framenum, mice_i * (2 * self.nkeypoints) + mice_j, :])
                          
        return retNodes, retEdges, retNodePresent, retEdgePresent




class ST_NODE():

    def __init__(self, node_type, node_id, node_pos_list):
        '''
        Initializer function for the ST node class
        params:
        node_type : Type of the node (Human or Obstacle)
        node_id : Pedestrian ID or the obstacle ID
        node_pos_list : Positions of the entity associated with the node in the sequence
        '''
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list = node_pos_list

    def getPosition(self, index):
        '''
        Get the position of the node at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.node_pos_list)
        return self.node_pos_list[index]

    def getType(self):
        '''
        Get node type
        '''
        return self.node_type

    def getID(self):
        '''
        Get node ID
        '''
        return self.node_id

    def addPosition(self, pos, index):
        '''
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.node_pos_list)
        self.node_pos_list[index] = pos

    def printNode(self):
        '''
        Print function for the node
        For debugging purposes
        '''
        print('Node type:', self.node_type, 'with ID:', self.node_id, 'with positions:', self.node_pos_list.values(), 'at time-steps:', self.node_pos_list.keys())


class ST_EDGE():

    def __init__(self, edge_type, edge_id, edge_pos_list):
        '''
        Inititalizer function for the ST edge class
        params:
        edge_type : Type of the edge (Human-Human or Human-Obstacle)
        edge_id : Tuple (or set) of node IDs involved with the edge
        edge_pos_list : Positions of the nodes involved with the edge
        '''
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.edge_pos_list = edge_pos_list

    def getPositions(self, index):
        '''
        Get Positions of the nodes at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.edge_pos_list)
        return self.edge_pos_list[index]

    def getType(self):
        '''
        Get edge type
        '''
        return self.edge_type

    def getID(self):
        '''
        Get edge ID
        '''
        return self.edge_id

    def addPosition(self, pos, index):
        '''
        Add a position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.edge_pos_list)
        self.edge_pos_list[index] = pos

    def printEdge(self):
        '''
        Print function for the edge
        For debugging purposes
        '''
        print('Edge type:', self.edge_type, 'between nodes:', self.edge_id, 'at time-steps:', self.edge_pos_list.keys())