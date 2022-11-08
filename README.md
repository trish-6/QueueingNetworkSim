# Queueing Network Simulation

- Author: Tristan Langley
- Date Created: 11/7/22
- Class: ECE 6101 AU22

## Overview
Includes a single python file simulating a queuing network with no feedback. Packets enter the system
according to a Poisson Process with rate lambda. They are routed through node 1, then through one of
the N nodes at the next level, then through node 3, after which they exit the network. Each node is 
modelled as a queue with exponential service time.

## Current phase
Implemented packet generator and each node (queue). Each of these runs as its own thread.

## Next Steps
- Try a larger simulation, recording delay and average # packets per queue
- Clarify questions on service time, link capacities, and routing to level 2 nodes

## Usage
- Tested on Python 3.10.0
- Command line `python proj.py`
