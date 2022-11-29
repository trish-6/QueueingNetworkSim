# Author: Tristan Langley
# Date created: 11/7/22
# Summary: Simulate a simple queueing network
#          - Packets arrive according to a Poisson process
#          - Packets go through node 1 -> one of N level 2 nodes -> node 3 -> exit system
#          - Link capacity C after nodes 1 and 3, link capacity C/N after nodes (2,n)
#          - Packet lengths are exponentially distributed
#          - Propagation delays and processing delays are negligible
# Usage: 'python proj.py', must have matplotlib installed


# --------------------------------------- Imports ----------------------------------------
import random
import threading
from collections import deque
import matplotlib.pyplot as plt
from typing import Type, Union, List, Tuple
# ----------------------------------------------------------------------------------------



# ----------------------------------- Global variables -----------------------------------
# EDIT THESE VALUES PER SIMULATION:
# How many level 2 nodes do you want? 1, 2, 3, or 4?
N = 4
# How many packets should pass through the system until stopping? Final sim is 50,000
END_PKTS = 50000

# Arrival rates (lambda values) to simulate
LAMBDAS = range(50, 1250, 50)
# Average packet length = 1000 bytes/pkt
PKT_LEN = 1000
# Link capacity of link1 and link3 is C = 10 Mbits/s = 1,250,000 bytes/s
C = 1250000
# Link capacity of level 2 links is C/N
CN = 1250000 / N
# ----------------------------------------------------------------------------------------



# --------------------------------------- Classes ----------------------------------------
# A single node (queue)
class Node:
    def __init__(self, id: int) -> None:
        # ID is either node 1, 2X, or 3 (where X depends on N, number of level 2 nodes)
        self.id = id
        # The pkts that are waiting in my queue; each value is the pkt's length in bytes
        self.queue = deque([])
        # The last packet that departed this node
        self.prev_pkt = [0, 0]
        # List of arrival times and departure times
        # The arrival and departure times of a packet have the same index in both lists
        # e.g. first pkt's arrival is self.arrivals[0], departure is self.departures[0]
        self.arrivals = []
        self.departures = []
        # The total number of pkts that have traveled through me
        self.total_pkts = 0

    # Add a received pkt to my queue
    def enqueue(self, pkt: List[float]) -> None:
        self.queue.append(pkt)
        self.arrivals.append(pkt[1])

    # 'Peek' at the departure time of the pkt at the front of the queue
    def next_dep(self) -> Union[List[float], None]:
        if len(self.queue) > 0:
            return max(self.queue[0][1], self.prev_pkt[1]) + self.queue[0][0] / CN
        else:
            return None
        
    # Send the pkt at the front of my queue to the next node provided.
    def send(self, next_node: 'Node') -> None:
        if len(self.queue) > 0:
            # Remove the pkt to send from this node's queue
            removed_pkt = self.queue.popleft()
            # Calculate: my departure time = max(my arr time, last dep time) + service time
            # service time = transmission delay = pkt length / link capacity
            if self.id == 1 or self.id == 3:
                dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / C
            else:
                dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / CN
            self.departures.append(dep_time)
            # Update the pkt's current time to this departure time
            removed_pkt[1] = dep_time
            # Update this node's last/previously departed pkt
            self.prev_pkt = removed_pkt[:]
            # Increment total packets that have passed through this node
            self.total_pkts += 1
            # Put the pkt into the next node's queue
            next_node.enqueue(removed_pkt)
    
    # Remove the pkt at the front of my queue; it exits the system and is gone for good
    # Requirement: only call this on node3 with link capacity C
    def remove(self) -> None:
        if len(self.queue) > 0:
            # Remove a pkt from this node's queue
            removed_pkt = self.queue.popleft()
            # Calculate: my departure time = max(my arr time, last dep time) + service time
            # service time = transmission delay = pkt length / link capacity
            dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / C
            self.departures.append(dep_time)
            # Update the pkt's current time to this departure time
            removed_pkt[1] = dep_time
            # Update this node's last/previously departed pkt
            self.prev_pkt = removed_pkt[:]
            # Increment total packets that have passed through this node
            self.total_pkts += 1
# ----------------------------------------------------------------------------------------



# -------------------------------------- Functions ---------------------------------------
# Pkt generator function: generates pkts according to a poisson process with 
# rate l and sends to node1. Runs for the number of minutes specified
def gen_pkts(node1: Node, node3: Node, l: int) -> None:
    cur_time = 0   # Time the current pkt is generated/arrives at node1
    while node3.total_pkts <= END_PKTS:
        # Distribution of time between successive pkts is exponential(l)
        cur_time += random.expovariate(l)
        # Generate the next pkt (length is exponential(1/1000))
        # pkt is a list of 2 items: [pkt length, current arrival time]
        cur_pkt = []
        cur_pkt.append(random.expovariate(1/PKT_LEN))  # pkt length
        cur_pkt.append(cur_time)                       # time of generation/arrival at node1
        # Put the pkt into node1's queue
        node1.enqueue(cur_pkt)


# The node1 function sends pkts to one of the N "level 2" nodes
def node1_func(node1: Node, node2_list: List[Node], node3: Node) -> None:
    while node3.total_pkts <= END_PKTS:
        # Check if there is a pkt in my queue
        if len(node1.queue) > 0:
            # Randomly choose one of the level 2 nodes to send the packet to
            node2_idx = random.randint(0, N-1)
            # Send the packet to that node
            node1.send(node2_list[node2_idx])

# The level2 function controls all N level 2 nodes
# Send the packet with the soonest departure time to node3
def level2_func(node2_list: List[Node], node3: Node) -> None:
    while node3.total_pkts <= END_PKTS:
        # Determine which level 2 node has the packet with the soonest departure time
        dep_times_None = [node.next_dep() for node in node2_list]
        dep_times = [x for x in dep_times_None if x != None]
        if len(dep_times) > 0:
            min_dep_node_idx = dep_times_None.index(min(dep_times))
            # Send pkt from that level 2 node to node3
            node2_list[min_dep_node_idx].send(node3)
    

# The node3 function removes pkts from node3; they are not sent anywhere, they exit the system
def node3_func(node3: Node) -> None:
    while node3.total_pkts <= END_PKTS:
        # Check if there is a pkt in my queue
        if len(node3.queue) > 0:
            # Remove pkt from my queue
            node3.remove()
# ----------------------------------------------------------------------------------------



# ---------------- Functions for a single simulation (one lambda value) ------------------
# Runs the simulation with the provided pkt arrival rate lambda (l)
# Returns two 3-tuples:
#     - The avg number of pkts in nodes 1, 20, and 3
#     - The avg delay in nodes 1, 20, and 3
def run_one_sim(l: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
   # Initialize nodes
    node1 = Node(1)
    node2_list = []
    for i in range(N):
        node2_list.append(Node(int('2' + str(i))))
    node3 = Node(3)
    
    # Run a thread that inserts packets into node1 following a P.P.
    # Last arg is lambda (arrival rate)
    pkt_gen_thread = threading.Thread(target=gen_pkts, name='PktGen',
                                      args=(node1, node3, l))
    pkt_gen_thread.start()
    
    # Create threads to control service and routing at each of the nodes
    # node1 thread
    node1_thread = threading.Thread(target=node1_func, name='Node1',
                                    args=(node1, node2_list, node3))
    node1_thread.start()
    # thread for level 2 nodes
    level2_thread = threading.Thread(target=level2_func, name='Nodes2x',
                                    args=(node2_list, node3))
    level2_thread.start()
    # node3 thread
    node3_thread = threading.Thread(target=node3_func, name='Node3', args=(node3,))
    node3_thread.start()
    
    # Wait for all threads to stop
    pkt_gen_thread.join()
    node1_thread.join()
    level2_thread.join()
    node3_thread.join()
    
    # Return the avg # of pkts in each node and the avg delay in each node, as 3-tuples
    return calc_avgs(node1, node2_list[0], node3)


# Calculates the average number of pkts and the average delays in nodes 1, 20, and 3
# Returns two 3-tuples:
#     - The avg number of pkts in nodes 1, 20, and 3
#     - The avg delay in nodes 1, 20, and 3
def calc_avgs(node1: Node, node2: Node, node3: Node) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # Sum up the total delays in each node
    sum_delays_1 = 0
    sum_delays_2 = 0
    sum_delays_3 = 0
    for i in range(len(node1.departures)):
        sum_delays_1 += node1.departures[i] - node1.arrivals[i]
    for i in range(len(node2.departures)):
        sum_delays_2 += node2.departures[i] - node2.arrivals[i]
    for i in range(len(node3.departures)):
        sum_delays_3 += node3.departures[i] - node3.arrivals[i]
    
    # Calculate the average number of packets in each node
    avg_pkts_1 = sum_delays_1 / (node1.departures[-1])
    avg_pkts_2 = sum_delays_2 / (node2.departures[-1])
    avg_pkts_3 = sum_delays_3 / (node3.departures[-1])
    
    # Calculate the average delays in each node
    avg_delay_1 = sum_delays_1 / node1.total_pkts
    avg_delay_2 = sum_delays_2 / node2.total_pkts
    avg_delay_3 = sum_delays_3 / node3.total_pkts
    
    # Return the avg # of pkts in each node and the avg delay in each node, as 3-tuples
    return (avg_pkts_1, avg_pkts_2, avg_pkts_3), (avg_delay_1, avg_delay_2, avg_delay_3)
# ----------------------------------------------------------------------------------------



# --------------------- Functions for calculating/plotting averages ----------------------
# Top-level function to calculate theoretical avgs and plot them with simulated avgs
def calc_plt_avgs(avg_pkts: List[float], avg_delays: List[float]) -> None:
    # Calculate theoretical averages by assuming each node is an M/M/1 queue
    # E(n) = expected # pkts in node = rho/(1-rho)
    mu13 = C / PKT_LEN
    mu2 = CN / PKT_LEN
    rho13 = [l / mu13 for l in LAMBDAS]
    rho2 = [(l/N) / mu2 for l in LAMBDAS]
    exp_pkts_13 = [r / (1 - r) for r in rho13]
    exp_pkts_2 = [r / (1 - r) for r in rho2]
    # Expected time spent in node = E(n) / lambda
    exp_time_13 = [n / l for n,l in zip(exp_pkts_13, LAMBDAS)]
    exp_time_2 = [n / (l/N) for n,l in zip(exp_pkts_2, LAMBDAS)]
    
    # Restructure the simulated data so it is easier to plot
    # Lists of avg # pkts in nodes, in order of lambda value
    node1_pkts = [item[0] for item in avg_pkts]
    node2_pkts = [item[1] for item in avg_pkts]
    node3_pkts = [item[2] for item in avg_pkts]
    # Lists of avg delays in nodes, in order of lambda value
    node1_delays = [item[0] for item in avg_delays]
    node2_delays = [item[1] for item in avg_delays]
    node3_delays = [item[2] for item in avg_delays]
    
    # Plot all these lists vs. lambda
    create_plots(node1_pkts, node2_pkts, node3_pkts, node1_delays, node2_delays,
                 node3_delays, exp_pkts_13, exp_pkts_2, exp_time_13, exp_time_2)
    

# Plot final averages, simulated and theoretical
def create_plots(p1: List[float], p2: List[float], p3: List[float], d1: List[float],
                 d2: List[float], d3: List[float], en13: List[float], en2: List[float],
                 et13: List[float], et2: List[float]) -> None:
    fig, axs = plt.subplots(3, 2)
    # Node 1 avg # pkts
    axs[0, 0].plot(LAMBDAS, p1)
    axs[0, 0].plot(LAMBDAS, en13)
    axs[0, 0].set_title('Node 1')
    axs[0, 0].set_xlabel('Lambda (pkts/s)')
    axs[0, 0].set_ylabel('Avg # of pkts')
    axs[0, 0].legend(labels=['Simulated','Theoretical'], loc='upper left')
    # Node 2 avg # pkts
    axs[1, 0].plot(LAMBDAS, p2)
    axs[1, 0].plot(LAMBDAS, en2)
    axs[1, 0].set_title('Node (2,n)')
    axs[1, 0].set_xlabel('Lambda (pkts/s)')
    axs[1, 0].set_ylabel('Avg # of pkts')
    axs[1, 0].legend(labels=['Simulated','Theoretical'], loc='upper left')
    # Node 3 avg # pkts
    axs[2, 0].plot(LAMBDAS, p3)
    axs[2, 0].plot(LAMBDAS, en13)
    axs[2, 0].set_title('Node 3')
    axs[2, 0].set_xlabel('Lambda (pkts/s)')
    axs[2, 0].set_ylabel('Avg # of pkts')
    axs[2, 0].legend(labels=['Simulated','Theoretical'], loc='upper left')
    # Node 1 avg delay
    axs[0, 1].plot(LAMBDAS, d1)
    axs[0, 1].plot(LAMBDAS, et13)
    axs[0, 1].set_title('Node 1')
    axs[0, 1].set_xlabel('Lambda (pkts/s)')
    axs[0, 1].set_ylabel('Avg delay (s)')
    axs[0, 1].legend(labels=['Simulated','Theoretical'], loc='upper left')
    # Node 2 avg delay
    axs[1, 1].plot(LAMBDAS, d2)
    axs[1, 1].plot(LAMBDAS, et2)
    axs[1, 1].set_title('Node (2,n)')
    axs[1, 1].set_xlabel('Lambda (pkts/s)')
    axs[1, 1].set_ylabel('Avg delay (s)')
    axs[1, 1].legend(labels=['Simulated','Theoretical'], loc='upper left')
    # Node 3 avg delay
    axs[2, 1].plot(LAMBDAS, d3)
    axs[2, 1].plot(LAMBDAS, et13)
    axs[2, 1].set_title('Node 3')
    axs[2, 1].set_xlabel('Lambda (pkts/s)')
    axs[2, 1].set_ylabel('Avg delay (s)')
    axs[2, 1].legend(labels=['Simulated','Theoretical'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------------



# ------------------------------------ Main function -------------------------------------
def main():
    # Lists for average # pkts in nodes 1, 20, and 3, and avg delay in nodes 1, 20, and 3
    # Structure: lists of 3-tuples
    # [ (_, _, _) , (_, _, _) , ... , (_, _, _) ]
    #     l=50        l=100     ...    l=1200
    avg_pkts_list = []
    avg_delays_list = []
    
    # Run simulation for each of lambda (arrival rate) = 50, 100, 150, ..., 1150, 1200
    for l in LAMBDAS:
        print('-------------------------- SIMULATION', l, '--------------------------')
        # Run a simulation with this lambda (pkt arrival rate)
        pkts, delays = run_one_sim(l)
        # Store avg number of pkts and avg delays in the lists
        avg_pkts_list.append(pkts)
        avg_delays_list.append(delays)
    
    # Plot the simulated and theoretical averages for each lambda
    calc_plt_avgs(avg_pkts_list, avg_delays_list)

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
