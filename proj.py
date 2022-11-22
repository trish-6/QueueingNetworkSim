## Steps:
# - Sender sends packets at lambda pkts/s
# - pkt goes from sender into node1
# - pkt goes through node1 and onto link1 with capacity C
# - pkt goes to one of n nodes with probability 1/n
# - pkt goes through node
# - pkt goes onto a link with capacity C/n into node3
# - pkt goes through node3
# - pkt goes onto link3 with capacity C
# - pkt exits system (arrives at destination node)


## Notes:
# - Pkts from sender follow a Poisson process with rate lambda
# - Pkt lengths are exponentially distributed (i.e. service times are exponentially distributed)
# - Propagation & processing delays are negligible (=> only consider queueing & transmission delay)


# Instructions:
# - Simulate for n = 1,2,3,4
# - lambda should range from {50, 100, 150, 200, ... 1100, 1150, 1200}
# - C = 10 Mbits/s = 1,250,000 bytes/s
# - Average pkt length is 1000 bytes
# - Keep track of 1. avg pkt delay through each node (queue)
#                 2. avg number of pkts in each node (queue)
# - Plot these avgs compared against theoretical avgs using Kleinrock's approx.


# -------------------------------------- Imports ---------------------------------------
import random
import threading
from collections import deque
import matplotlib.pyplot as plt
from typing import Type, Union, List, Tuple
# --------------------------------------------------------------------------------------


# ---------------------------------- Global variables ----------------------------------
# EDIT THESE VALUES PER SIMULATION:
# How many level 2 nodes do you want? 1, 2, 3, or 4?
N = 2
# How many packets should pass through the system until I stop? for now, 5
END_PKTS = 5

# Link capacity of link1 and link3 is C = 10 Mbits/s = 1,250,000 bytes/s
C = 1250000
# Link capacity of level 2 links is C/N
CN = 1250000 / N
# --------------------------------------------------------------------------------------


# -------------------------------------- Classes ---------------------------------------
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
        if self.id == 1:
            print('ENTERED SYSTEM: PKT %.0f AT TIME %.3f' %(pkt[0], pkt[1]))
        
    # Send the pkt at the front of my queue to the next node provided.
    def send(self, next_node: 'Node') -> None:
        if len(self.queue) > 0:
            # Remove the pkt to send from this node's queue
            removed_pkt = self.queue.popleft()
            # FOR TESTING ONLY
            my_arr_time = removed_pkt[1]
            # Calculate: departure time = max(my arr time, last dep time) + service time
            # service time = transmission delay = pkt length / link capacity
            if self.id == 1 or self.id == 3:
                dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / C
            else:
                dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / CN
            self.departures.append(dep_time)
            # Update the pkt's current time to this departure time
            removed_pkt[1] = dep_time
            # Put the pkt into the next node's queue
            next_node.enqueue(removed_pkt)
            # Update this node's last/previously departed pkt
            self.prev_pkt = removed_pkt
            # Increment total packets that have passed through this node
            self.total_pkts += 1
            print('%.0f %d->%d DELAY %.7f/%.7f AT TIME %.3f' 
                  %(removed_pkt[0], self.id, next_node.id, self.departures[-1] - self.arrivals[len(self.departures)-1],
                    dep_time - my_arr_time, removed_pkt[1]))
    
    # Remove the pkt at the front of my queue; it exits the system and is gone for good
    # Requirement: only call this on node3 with link capacity C
    def remove(self) -> None:
        if len(self.queue) > 0:
            # Remove a pkt from this node's queue
            removed_pkt = self.queue.popleft()
            # FOR TESTING ONLY
            my_arr_time = removed_pkt[1]
            # Calculate: departure time = max(my arr time, last dep time) + service time
            # service time = transmission delay = pkt length / link capacity
            dep_time = max(removed_pkt[1], self.prev_pkt[1]) + removed_pkt[0] / C
            self.departures.append(dep_time)
            # Update the pkt's current time to this departure time
            removed_pkt[1] = dep_time
            # Update this node's last/previously departed pkt
            self.prev_pkt = removed_pkt
            # Increment total packets that have passed through this node
            self.total_pkts += 1
            print('EXIT %.0f DELAY %.7f/%.7f AT TIME %.3f'
                  %(removed_pkt[0], self.departures[-1] - self.arrivals[len(self.departures)-1],
                    dep_time - my_arr_time, removed_pkt[1]))
# --------------------------------------------------------------------------------------


# ------------------------------------- Functions --------------------------------------
# Pkt generator function: generates pkts according to a poisson process with 
# rate l and sends to node1. Runs for the number of minutes specified
def gen_pkts(node1: Node, node3: Node, l: int) -> None:
    cur_time = 0   # Time the current pkt is generated/arrives at node1
    while node3.total_pkts <= END_PKTS:
        # Distribution of time between successive pkts is exponential(l)
        cur_time += random.expovariate(l)
        # Generate the next pkt (length is exponential(1/1000))
        # pkt is a list of 3 items: [pkt length, time I was sent, index]
        cur_pkt = []
        cur_pkt.append(random.expovariate(0.001))  # pkt length
        cur_pkt.append(cur_time)                   # time of generation/arrival at node1
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
        

# The node2 function sends pkts to node3
def node2_func(node2: Node, node3: Node) -> None:
    while node3.total_pkts <= END_PKTS:
        # Check if there is a pkt in my queue
        if len(node2.queue) > 0:
            # Send pkt to node3
            node2.send(node3)
        

# The node3 function removes pkts from node3; they are not sent anywhere, they exit the system
def node3_func(node3: Node) -> None:
    while node3.total_pkts <= END_PKTS:
        # Check if there is a pkt in my queue
        if len(node3.queue) > 0:
            # Remove pkt from my queue
            node3.remove()
# --------------------------------------------------------------------------------------


# --------------- Functions for a single simulation (one lambda value) -----------------
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
    
    # Create threads for each of the nodes
    # Each thread handles sending the pkt at the front of the queue to the next node
    # node2 thread
    node1_thread = threading.Thread(target=node1_func, name='Node1',
                                    args=(node1, node2_list, node3))
    node1_thread.start()
    # threads for level 2 nodes
    node2_threads = []
    for i in range(N):
        node2_threads.append(threading.Thread(target=node2_func, name='Node2'+str(i),
                             args=(node2_list[i], node3)))
        node2_threads[i].start()
    # node3 thread
    node3_thread = threading.Thread(target=node3_func, name='Node3', args=(node3,))
    node3_thread.start()
    
    # Wait for all threads to stop
    pkt_gen_thread.join()
    node1_thread.join()
    for i in range(N):
        node2_threads[i].join()
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
        print('Node1 delay: %.7f' %(node1.departures[i] - node1.arrivals[i]))
        sum_delays_1 += node1.departures[i] - node1.arrivals[i]
    for i in range(len(node2.departures)):
        print('Node2 delay: %.7f' %(node2.departures[i] - node2.arrivals[i]))
        sum_delays_2 += node2.departures[i] - node2.arrivals[i]
    for i in range(len(node3.departures)):
        print('Node3 delay: %.7f' %(node3.departures[i] - node3.arrivals[i]))
        sum_delays_3 += node3.departures[i] - node3.arrivals[i]
    
    print('Node1 total pkts: %d' %node1.total_pkts)
    print('Node2 total pkts: %d' %node2.total_pkts)
    print('Node3 total pkts: %d' %node3.total_pkts)
    
    print('Node1 total time: %.3f' %(node1.departures[-1]))
    print('Node2 total time: %.3f' %(node2.departures[-1]))
    print('Node3 total time: %.3f' %(node3.departures[-1]))
    
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
    
# --------------------------------------------------------------------------------------


# ----------------------------------- Main function ------------------------------------
def main():
    # Lists for average # pkts in nodes 1, 20, and 3, and avg delay in nodes 1, 20, and 3
    # Structure: lists of 3-tuples
    # [ (_, _, _) , (_, _, _) , ... , (_, _, _) ]
    #     l=50        l=100     ...    l=1200
    avg_pkts_list = []
    avg_delays_list = []
    
    # Run simulation for each of lambda (arrival rate) = 50, 100, 150, ..., 1150, 1200
    lambdas = range(1, 2, 1)
    for l in lambdas:
        print('-------------------------- SIMULATION', l, '--------------------------')
        # Run a simulation with this lambda (pkt arrival rate)
        pkts, delays = run_one_sim(l)
        # Store avg number of pkts and avg delays in the lists
        avg_pkts_list.append(pkts)
        avg_delays_list.append(delays)
    
    # Restructure the data so it is easier to plot
    # Lists of avg # pkts in nodes, in order of lambda value
    node1_pkts = [item[0] for item in avg_pkts_list]
    node2_pkts = [item[1] for item in avg_pkts_list]
    node3_pkts = [item[2] for item in avg_pkts_list]
    # Lists of avg delays in nodes, in order of lambda value
    node1_delays = [item[0] for item in avg_delays_list]
    node2_delays = [item[1] for item in avg_delays_list]
    node3_delays = [item[2] for item in avg_delays_list]
    
    # Plot final statistics
    fig, axs = plt.subplots(3, 2)
    # Node 1 avg # pkts
    axs[0, 0].scatter(lambdas, node1_pkts)
    axs[0, 0].set_title('Node 1')
    axs[0, 0].set_xlabel('Lambda (pkts/s)')
    axs[0, 0].set_ylabel('Avg # of pkts')
    # Node 2 avg # pkts
    axs[1, 0].scatter(lambdas, node2_pkts)
    axs[1, 0].set_title('Node (2,n)')
    axs[1, 0].set_xlabel('Lambda (pkts/s)')
    axs[1, 0].set_ylabel('Avg # of pkts')
    # Node 3 avg # pkts
    axs[2, 0].scatter(lambdas, node3_pkts)
    axs[2, 0].set_title('Node 3')
    axs[2, 0].set_xlabel('Lambda (pkts/s)')
    axs[2, 0].set_ylabel('Avg # of pkts')
    # Node 1 avg delay
    axs[0, 1].scatter(lambdas, node1_delays)
    axs[0, 1].set_title('Node 1')
    axs[0, 1].set_xlabel('Lambda (pkts/s)')
    axs[0, 1].set_ylabel('Avg delay (s)')
    # Node 2 avg delay
    axs[1, 1].scatter(lambdas, node2_delays)
    axs[1, 1].set_title('Node (2,n)')
    axs[1, 1].set_xlabel('Lambda (pkts/s)')
    axs[1, 1].set_ylabel('Avg delay (s)')
    # Node 3 avg delay
    axs[2, 1].scatter(lambdas, node3_delays)
    axs[2, 1].set_title('Node 3')
    axs[2, 1].set_xlabel('Lambda (pkts/s)')
    axs[2, 1].set_ylabel('Avg delay (s)')
    
    plt.tight_layout()
    plt.show()
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
