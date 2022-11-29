# Queueing Network Simulation

- Author: Tristan Langley
- Date Created: 11/7/22
- Class: ECE 6101 AU22

## Overview
Includes a single python file simulating a queuing network with no feedback. Packets enter the system
according to a Poisson process with rate lambda. They are routed through node 1, then through one of
the N nodes at the next level, then through node 3, after which they exit the network. Each node is 
modeled as a queue with exponential service time.

## Basic Implementation
Each node is an instance of a class called 'Node', which keeps track of information like the node's ID
(e.g. 1, 20, 21, 3), a list of packets in the node's queue, and arrival/departure times of packets.
The program uses one thread for each 'level' of nodes, which run in parallel. Specifically, there is one
thread for node 1, one thread for the N level 2 nodes, and one thread for node 3. Each thread handles the
arrivals, services, and departures of packets in that node level. Once a certain number of packets pass
through the system (e.g. 50,000), the simulation ends and the program plots average values for number of
packets in each node and delay through each node.

## Current Phase
Implemented the entire queueing system, and calculation/plotting of average number of packets and
average delay through each node. Manually checked for correct output with a slow arrival rate and a
low number of packets. With higher arrival rates and number of packets, plots seem reasonable from
comparing theoretical and simulated values.

## Next Steps
- Simulate 50,000 packets for each of N = 1,2,3,4 and compare simulated vs. theoretical results

## Requirements and Usage
- Matplotlib required
- Tested on Python 3.10.0
- Command line `python proj.py`
