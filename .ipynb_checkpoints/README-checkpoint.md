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
(e.g. 1, 20, 21, 3), a list of packets in the node's queue, and arrival/departure times of packets.
The program uses one thread for each node, which run in parallel. Each thread handles the arrivals,
services, and departures of packets in that node.

## Current Phase
Implemented the entire queueing system and calculation of average number of packets and average delay
through each node. The main function runs one simulations with lambda = 1 pkt/s, and plots the simulated
averages for # pkts and delay in each node. Thorough logging of packet movement by printing to console.
Manually checked for correct output.

## Next Steps
- Try more/larger simulations
- Calculate and plot theoretical values based on Kleinrock's Independence Approx.
- Update the look of the graphs: line vs. scatter, fixed axis ranges, etc.

## Requirements and Usage
- Matplotlib required
- Tested on Python 3.10.0
- Command line `python proj.py`
