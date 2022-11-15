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
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------


# ---------------------------------- Global variables ----------------------------------
# EDIT THESE VALUES PER SIMULATION:
# How many level 2 nodes do you want? 1, 2, 3, or 4?
N = 2
# How long should the simulation last, in minutes? TODO for now, 5 seconds
DURATION = 0.08333

# Link capacity of link1 and link3 is C = 10 Mbits/s = 1,250,000 bytes/s
C = 1250000
# Link capacity of level 2 links is C/N
CN = 1250000 / N
# --------------------------------------------------------------------------------------


# -------------------------------------- Classes ---------------------------------------
# A single node (queue)
class Node:
    def __init__(self, id):
        # ID is either node 1, 2x, or 3 (where x depends on N, number of level 2 nodes)
        self.id = id
        # The pkts that are waiting in my queue; each value is the pkt's length in bytes
        self.queue = deque([])
        # The total number of pkts that have traveled through me
        # Used for: total pkts / total simulation time = avg # pkts at any given moment
        self.total_pkts = 0
        # The total amount of time that packets have spent waiting in my queue
        # TODO figure out how to keep track of delay
        self.total_waits = 0
    
    # 'Peek' at the next pkt to be removed from the queue, without actually removing it
    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            return None

    # Add a received pkt to my queue
    def enqueue(self, pkt):
        self.queue.append(pkt)
        self.total_pkts += 1
        #if self.id == 1:
            #print('ENTERED SYSTEM: PKT', int(pkt))
        
    # Send the pkt at the front of my queue to the next node provided
    def send(self, next_node):
        if len(self.queue) > 0:
            removed_pkt = self.queue.popleft()
            next_node.enqueue(removed_pkt)
            # TODO this total waits calculation is wrong. this gets total service time, not total wait time
            self.total_waits += removed_pkt / 1000
            #print('PKT', int(removed_pkt), 'SENT FROM NODE', self.id, '-> NODE', next_node.id)
    
    # Remove the pkt at the front of my queue; it exits the system and is gone for good
    def remove(self):
        if len(self.queue) > 0:
            removed_pkt = self.queue.popleft()
            # TODO this total waits calculation is wrong. this gets total service time, not total wait time
            self.total_waits += removed_pkt / 1000
            #print('EXITED SYSTEM: PKT', int(removed_pkt))
# --------------------------------------------------------------------------------------


# ------------------------------------- Functions --------------------------------------
# Pkt generator function: generates pkts according to a poisson process with 
# rate l and sends to node1. Runs for the number of minutes specified
def gen_pkts(node1, l, t_end):
    while time.time() < t_end:
        # Generate the amount of time to wait until the next pkt is released
        # Distribution of time between successive pkts is exponential(l)
        wait_time = random.expovariate(l)
        # Wait for that amount of time
        time.sleep(wait_time)
        # Generate the next pkt (length is exponential(1/1000)) and send to node1
        node1.enqueue(random.expovariate(0.001))

    
# The node1 function sends pkts to one of the N "level 2" nodes periodically,
# depending on service time (pkt length)
def node1_func(node1, node2_list, t_end):
    while time.time() < t_end:
        # Check if there is a pkt in my queue
        pkt = node1.peek()
        if pkt != None:
            # Randomly choose one of the level 2 nodes to send the packet to
            node2_idx = random.randint(0, N-1)
            # Send the packet to that node after the amount of time dictated by its length
            # service time = transmission delay = pkt length / link capacity
            time.sleep(pkt / C)
            node1.send(node2_list[node2_idx])
        

# The node2 function sends pkts to node3 depending on service time (pkt length)
def node2_func(node2, node3, t_end):
    while time.time() < t_end:
        # Check if there is a pkt in my queue
        pkt = node2.peek()
        if pkt != None:
            # Send it to node3 after the amount of time dictated by its length
            # service time = transmission delay = pkt length / link capacity
            time.sleep(pkt / CN)
            node2.send(node3)
        

# The node3 function removes pkts from node3 depending on service time (pkt length)
# They are not sent anywhere; they exit the system
def node3_func(node3, t_end):
    while time.time() < t_end:
        # Check if there is a pkt in my queue
        pkt = node3.peek()
        if pkt != None:
            # Remove pkt from queue after the amount of time dictated by its length
            # service time = transmission delay = pkt length / link capacity
            time.sleep(pkt / C)
            node3.remove()
# --------------------------------------------------------------------------------------


# ------------ "Main" function for a single simulation (one lambda value) --------------
# Runs the simulation with the provided pkt arrival rate lambda (l)
# Returns two 3-tuples:
#     - The avg number of pkts in nodes 1, 20, and 3
#     - The avg delay in nodes 1, 20, and 3
def run_one_sim(l, t_end):
   # Initialize nodes
    node1 = Node(1)
    node2_list = []
    for i in range(N):
        node2_list.append(Node(int('2' + str(i))))
    node3 = Node(3)
    
    # Run a thread in the background that inserts packets into node1 following a P.P.
    # Second arg is lambda (arrival rate)
    pkt_gen_thread = threading.Thread(target=gen_pkts, name='PktGen', args=(node1, l, t_end))
    pkt_gen_thread.start()
    
    # Create threads for each of the nodes
    # Each thread handles sending the pkt at the front of the queue to the next node
    # node2 thread
    node1_thread = threading.Thread(target=node1_func, name='Node1', args=(node1, node2_list, t_end))
    node1_thread.start()
    # threads for level 2 nodes
    node2_threads = []
    for i in range(N):
        node2_threads.append( threading.Thread(target=node2_func, name='Node2'+str(i), \
                              args=(node2_list[i], node3, t_end)) )
        node2_threads[i].start()
    # node3 thread
    node3_thread = threading.Thread(target=node3_func, name='Node3', args=(node3, t_end))
    node3_thread.start()
    
    
    # Dummy code in foreground: print one 'x' every second
    while time.time() < t_end:
        time.sleep(1)
        print('x')
    
    
    # Wait for all threads to stop
    pkt_gen_thread.join()
    node1_thread.join()
    for i in range(N):
        node2_threads[i].join()
    node3_thread.join()
    
    # Calculate the average number of packets in each node
    avg_pkts_1 = node1.total_pkts / (DURATION * 60)
    avg_pkts_2 = node2_list[0].total_pkts / (DURATION * 60)
    avg_pkts_3 = node3.total_pkts / (DURATION * 60)
    
    # TODO dummy 0's for avg delays for now
    avg_delays = (0, 0, 0)
    
    # Return the avg # of pkts in each node and the avg delay in each node, as 3-tuples
    return (avg_pkts_1, avg_pkts_2, avg_pkts_3), avg_delays
    

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
    lambdas = range(5, 30, 5)
    for l in lambdas:
        print('------------------------------ SIMULATION', l, '------------------------------')
        # End time of this simulation
        t_end = time.time() + 60 * DURATION
        # Run a simulation with this lambda (pkt arrival rate)
        pkts, delays = run_one_sim(l, t_end)
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
    
    #print(node1_pkts)
    
    # Plot final statistics
    fig, axs = plt.subplots(3, 2)
    # Node 1 avg # pkts
    axs[0, 0].scatter(lambdas, node1_pkts)
    axs[0, 0].set_title('Node 1')
    axs[0, 0].set_xlabel('Lambda (pkts/s)')
    axs[0, 0].set_ylabel('Avg # of pkts')
    # Node 1 avg # pkts
    axs[1, 0].scatter(lambdas, node2_pkts)
    axs[1, 0].set_title('Node (2,n)')
    axs[1, 0].set_xlabel('Lambda (pkts/s)')
    axs[1, 0].set_ylabel('Avg # of pkts')
    # Node 1 avg # pkts
    axs[2, 0].scatter(lambdas, node3_pkts)
    axs[2, 0].set_title('Node 3')
    axs[2, 0].set_xlabel('Lambda (pkts/s)')
    axs[2, 0].set_ylabel('Avg # of pkts')
    
    plt.show()
    
    
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
