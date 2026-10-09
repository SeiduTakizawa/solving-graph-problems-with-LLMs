The shortest route through a waypoint is a shortest path to the waypoint, then a shortest path from it:
1. shortest_path(source, via) and shortest_path(via, target)
2. join them, with the waypoint once: first + second[1:]
Submit that list. A single shortest_path(source, target) usually skips the waypoint: don't use it.
