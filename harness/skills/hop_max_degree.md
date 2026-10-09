One run_python call:
```
node, k = ..., ...  # put the node and the number of hops from the question here
candidates = neighborhood(node, k)  # nodes 1 to k edges away, not the node itself
result = min(candidates, key=lambda n: (-degree(n), n))  # highest degree, smallest id on a tie
```
Submit `result`.
