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


## Questions:
# - What is the point of a link capacity C = 1,250,000 bytes/s if pkt lengths are 
#   only 1000 bytes, and propagation delay is ignored?
# - Service time is exponential and based on pkt length. How to translate this into an
#   actual number of seconds? Is it just a transmission delay calculation? i.e. does
#   service time == transmission delay?
#   Transmission delay = pkt length (bytes) * [1 / C] (s / byte)
# - How to best implement the 1/n probability for the level 2 nodes? exactly split? or, 
#   random yields 1/n "on average" as time -> infinity?


# -------------------------------------- Imports ---------------------------------------
import random
import time
import threading
from collections import deque
# --------------------------------------------------------------------------------------


# ---------------------------------- Global variables ----------------------------------
# EDIT THESE VALUES PER SIMULATION:
# How many level 2 nodes do you want? 1, 2, 3, or 4?
N = 3
# What is lambda (arrival rate)? TODO for now, let L = 1 pkt/s
L = 10
# How long should the simulation last, in minutes? TODO for now, 10 seconds
DURATION = 0.167

# End time of simulation
T_END = time.time() + 60 * DURATION
# Link capacity of link1 and link3 is C = 10 Mbits/s = 1,250,000 bytes/s
C = 1250000
# Link capacity of level 2 links is C/N
CN = 1250000 / N
# --------------------------------------------------------------------------------------


# -------------------------------------- Classes ---------------------------------------
# A single node (queue)
class Node:
    def __init__(self, id):
        self.id = id
        # The pkts that are waiting in my queue; each value is the pkt's length in bytes
        self.queue = deque([])
        # The total number of pkts that have traveled through me
        # Used for: total pkts / total simulation time = avg # pkts at any given moment
        self.total_pkts = 0
        # The total amount of time that packets have spent waiting in my queue
        # Used for: total time pkts spend in queue / total pkts = avg delay per pkt
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
        if self.id == 1:
            print('ENTERED SYSTEM: PKT', int(pkt))
        
    # Send the pkt at the front of my queue to the next node provided
    def send(self, next_node):
        if len(self.queue) > 0:
            removed_pkt = self.queue.popleft()
            next_node.enqueue(removed_pkt)
            self.total_waits += removed_pkt / 1000
            print('PKT', int(removed_pkt), 'SENT FROM NODE', self.id, '-> NODE', next_node.id)
    
    # Remove the pkt at the front of my queue; it exits the system and is gone for good
    def remove(self):
        if len(self.queue) > 0:
            removed_pkt = self.queue.popleft()
            self.total_waits += removed_pkt / 1000
            print('EXITED SYSTEM: PKT', int(removed_pkt))
# --------------------------------------------------------------------------------------


# ------------------------------------- Functions --------------------------------------
# Pkt generator function: generates pkts according to a poisson process with 
# rate l and sends to node1. Runs for the number of minutes specified
def gen_pkts(node1):
    while time.time() < T_END:
        # Generate the amount of time to wait until the next pkt is released
        # Distribution of time between successive pkts is exponential(l)
        wait_time = random.expovariate(L)
        # Wait for that amount of time
        time.sleep(wait_time)
        # Generate the next pkt (length is exponential(1/1000)) and send to node1
        node1.enqueue(random.expovariate(0.001))
        #print(time.time())

    
# The node1 function sends pkts to one of the N "level 2" nodes periodically,
# depending on service time (pkt length)
def node1_func(node1, node2_list):
    node2_idx = 0
    while time.time() < T_END:
        # Check if there is a pkt in my queue
        pkt = node1.peek()
        if pkt != None:
            # Send it to one of the level 2 nodes after the amount of time dictated by its length
            # TODO service time == transmission delay ?
            time.sleep(pkt / C)
            node1.send(node2_list[node2_idx])
        # TODO for now, just loop through the N level 2 nodes to decide which to send it to
        node2_idx = (node2_idx + 1) % N
        

# The node2 function sends pkts to node3 depending on service time (pkt length)
def node2_func(node2, node3):
    while time.time() < T_END:
        # Check if there is a pkt in my queue
        pkt = node2.peek()
        if pkt != None:
            # Send it to node3 after the amount of time dictated by its length
            # TODO service time == transmission delay ?
            time.sleep(pkt / CN)
            node2.send(node3)
        

# The node3 function removes pkts from node3 depending on service time (pkt length)
# They are not sent anywhere; they exit the system
def node3_func(node3):
    while time.time() < T_END:
        # Check if there is a pkt in my queue
        pkt = node3.peek()
        if pkt != None:
            # Remove pkt from queue after the amount of time dictated by its length
            # TODO service time == transmission delay ?
            time.sleep(pkt / C)
            node3.remove()
# --------------------------------------------------------------------------------------


# ----------------------------------- Main function ------------------------------------
def main():
    # Initialize nodes
    node1 = Node(1)
    node2_list = []
    for i in range(N):
        node2_list.append(Node(int('2' + str(i))))
    node3 = Node(3)
    
    # Run a thread in the background that inserts packets into node1 following a P.P.
    # Second arg is lambda (arrival rate)
    pkt_gen_thread = threading.Thread(target=gen_pkts, name='PktGen', args=(node1,))
    pkt_gen_thread.start()
    
    # Create threads for each of the nodes
    # Each thread handles sending the pkt at the front of the queue to the next node
    # node2 thread
    node1_thread = threading.Thread(target=node1_func, name='Node1', args=(node1, node2_list))
    node1_thread.start()
    # threads for level 2 nodes
    node2_threads = []
    for i in range(N):
        node2_threads.append( threading.Thread(target=node2_func, name='Node2'+str(i), \
                              args=(node2_list[i], node3)) )
        node2_threads[i].start()
    # node3 thread
    node3_thread = threading.Thread(target=node3_func, name='Node3', args=(node3,))
    node3_thread.start()
    
    
    # Dummy code in foreground: print one 'x' every second
    while time.time() < T_END:
        time.sleep(1)
        print('x')
    
    
    # Wait for all threads to stop
    pkt_gen_thread.join()
    node1_thread.join()
    for i in range(N):
        node2_threads[i].join()
    node3_thread.join()
    
    # Final statistics
    #print(node1.total_waits)
    #for i in range(N):
    #    print(node2_list[i].total_waits)
    #print(node3.total_waits)
# --------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
