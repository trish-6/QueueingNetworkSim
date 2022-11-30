# Queueing Network Simulation

- Author: Tristan Langley
- Date Created: 11/7/22
- Class: ECE 6101 AU22

## Overview
Includes a single python file simulating a queuing network with no feedback. Packets enter the system
according to a Poisson process with rate lambda. They are routed through node 1, then through one of
the N nodes at the next level, then through node 3, after which they exit the network. Each node is 
modeled as a queue with exponential service time.

## Output
The program outputs numerous plots of the average number of packets in each node, and the average packet
delay through each node, for each lambda and N. Both simulated and theoretical values are plotted and
compared. Theoretical values are calculated by assuming that each node is an M/M/1 queue.

## Basic Implementation
Each node is an instance of a class called 'Node', which keeps track of information like the node's ID
(e.g. 1, 20, 21, 3), a list of packets in the node's queue, and arrival/departure times of packets.
The program uses one thread for each 'level' of nodes, which run in parallel. Specifically, there is one
thread for node 1, one thread for the N level 2 nodes, and one thread for node 3. Each thread handles the
arrivals, services, and departures of packets in that node level. Once a certain number of packets pass
through the system (e.g. 50,000), the simulation ends and the program plots average values for number of
packets in each node and delay through each node.

## Requirements and Usage
- Matplotlib required
- Tested on Python 3.10.0
- Command line `python proj.py`
