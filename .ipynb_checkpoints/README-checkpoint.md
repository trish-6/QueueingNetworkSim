# Queueing Network Simulation

- Author: Tristan Langley
- Date Created: 11/7/22
- Class: ECE 6101 AU22

## Overview
Includes a single python file simulating a queuing network with no feedback. Packets enter the system
according to a Poisson Process with rate lambda. They are routed through node 1, then through one of
the N nodes at the next level, then through node 3, after which they exit the network. Each node is 
modelled as a queue with exponential service time.

## Basic Implementation
Each node is an instance of a class called 'Node', which keeps track of information like the node's ID
(e.g. 1, 20, 21, 3), a list of packets in the node's queue, and the total number of packets that have
been in the node. The program runs by threads for each node, which run in parallel. Each thread
handles the arrivals, services, and departures of packets in that node.

## Current phase
Implemented the entire queueing system, aside from keeping track of the packet delay through each
node. Main function runs a few simulations with different lambdas (arrival rates), and plots the
simulated averages for # pkts in each node.

## Next Steps
- Keep track of packet delay in each node
- Fix the plots so labels aren't overlapping
- Calculate and plot theoretical values based on Kleinrock's Independence Approx.
- Try larger simulations

## Requirements and Usage
- Matplotlib required
- Tested on Python 3.10.0
- Command line `python proj.py`
