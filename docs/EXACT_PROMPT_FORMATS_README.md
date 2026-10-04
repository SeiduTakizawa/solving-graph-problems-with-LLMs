# Exact Prompt Formats for All Problem Types and Techniques

This README provides the exact prompt formats used for every combination of problem type and prompting technique in the graph reasoning experiments.

## Base Question Format

All questions start with this base format:
```
Let G be a graph. The edgelist of graph G is the following: [(edge_list)]. [Question]
```

## Prompting Techniques

### 1. `none` (Baseline)
**Format**: `[Base Question] + [Output Format]`

### 2. `cot` (Chain of Thought)  
**Format**: `[Base Question] + [Output Format] + " Let's think step by step."`

### 3. `bag` (Build a Graph)
**Format**: `[Base Question] + [Output Format] + " Let's construct a graph with the nodes and edges first."`

### 4. `alg` (Algorithm)
**Format**: `[Pseudocode] + " " + [Base Question] + [Algorithm Output Format]`

### 5. `default_1_shot` (One-Shot Example)
**Format**: `[Example Problem + Solution] + " " + [Base Question] + [Output Format]`

### 6. `1_shot_pseudo` (One-Shot + Pseudocode)
**Format**: `[Pseudocode + Example Problem + Step-by-step Solution] + " " + [Base Question] + [Algorithm Output Format]`

---

## Problem-Specific Prompt Examples

### NODE COUNT

#### Base Question:
```
"What is the number of nodes of G?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: The number of nodes is ..."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of nodes is ..."`

#### Complete Prompts by Technique:

**`none`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Output the result like that: The number of nodes is ...
```

**`cot`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Output the result like that: The number of nodes is ... Let's think step by step.
```

**`bag`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Output the result like that: The number of nodes is ... Let's construct a graph with the nodes and edges first.
```

**`alg`:**
```
Function CountNodes(EdgeList):
    // EdgeList is a list where each element is a tuple (u, v),
    // indicating an edge between nodes u and v.

    // Initialize a set to store unique nodes
    Found = Set()

    // Iterate through each edge in the edge list
    For each edge (u, v) in EdgeList:
        // Add both nodes of the edge to the set
	if u not in Found:
            Add u to Found
        if u not in Found:
            Add v to Found

    // The size of Found represents the total number of unique nodes
    Return Size of Found

Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of nodes is ...
```

**`default_1_shot`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (3, 5), (2, 5), (4, 5)]. What is the number of nodes of G? Output the result like that: The number of nodes is ...
To find the number of nodes in graph G, we can observe the unique vertices present in the edge list. Let's count them:

(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (3, 5), (2, 5), (4, 5)

The unique vertices are: 0, 1, 2, 3, 4, 5.

Therefore, the number of nodes in graph G is 6.

Output: "The number of nodes is 6."

Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Output the result like that: The number of nodes is ...
```

**`1_shot_pseudo`:**
```
[Pseudocode + Example + Step-by-step execution] + Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of nodes of G? Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of nodes is ...
```

---

### EDGE COUNT

#### Base Question:
```
"What is the number of edges of G?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: The number of edges is ..."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of edges is ..."`

#### Complete Prompts by Technique:

**`none`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of edges of G? Output the result like that: The number of edges is ...
```

**`cot`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of edges of G? Output the result like that: The number of edges is ... Let's think step by step.
```

**`default_1_shot`:**
```
Let G be a graph. The edgelist of graph G is the following: [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (3, 5), (2, 5), (4, 5)]. What is the number of edges of G? Output the result like that: The number of edges is ...

The number of edges is 8.

Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. What is the number of edges of G? Output the result like that: The number of edges is ...
```

---

### CYCLE CHECK

#### Base Question:
```
"Is there a cycle in G?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: \"There is a cycle\" or \"There is no cycle.\""`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result like that: \"There is a cycle\" or \"There is no cycle.\""`

#### Complete Prompts by Technique:

**`1_shot_pseudo` (Example):**
```
function hasCycle(graph):
    visited = set()

    for node in graph:
        if node not in visited:
            if dfs(node, -1, visited, graph) == True:
                return "There is a cycle"

    return "There is no cycle"

function dfs(current, parent, visited, graph):
    mark current as visited
    for neighbor in graph[current]:
        if neighbor is not visited:
            if dfs(neighbor, current, visited, graph) == True:
                return True
        else if neighbor != parent:
            return True
    return False


Let G be a graph. The edge list of graph G is the following:
[(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (3, 5), (2, 5), (4, 5)]

Is there a cycle in the graph G?

To determine whether there is a cycle in this graph, apply the Depth-First Search (DFS) algorithm with cycle detection. Follow the provided pseudocode step-by-step and show all steps clearly, including:

Step-by-step execution:

1. Build the graph as an adjacency list from the edgelist:

graph = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 5],
    4: [1, 3, 5],
    5: [3, 2, 4]
}

2. Initialize `visited = {}`

3. Start with node 0 (not visited):
   Call `dfs(0, -1)`
   → Mark 0 visited → visited = {0}
   → Visit neighbor 1 → call `dfs(1, 0)`
   → Mark 1 visited → visited = {0, 1}
   → Neighbor 0 is parent → skip
   → Visit neighbor 2 → call `dfs(2, 1)`
   → Mark 2 visited → visited = {0, 1, 2}
   → Neighbor 1 is parent → skip
   → Visit neighbor 5 → call `dfs(5, 2)`
   → Mark 5 visited → visited = {0, 1, 2, 5}
   → Visit neighbor 3 → call `dfs(3, 5)`
   → Mark 3 visited → visited = {0, 1, 2, 3, 5}
   → Visit neighbor 0 → already visited and not parent → **cycle found**
Output: There is a cycle.

Let G be a graph. The edgelist of graph G is the following: [(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]. Is there a cycle in G? Follow the provided pseudocode step-by-step and show all steps. Output the result like that: "There is a cycle" or "There is no cycle."
```

---

### CONNECTED COMPONENTS COUNT

#### Base Question:
```
"What is the number of connected components of G?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: The number of connected components is..."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of connected components is ..."`

---

### BIPARTITE CHECK

#### Base Question:
```
"Is graph G bipartite or not?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: \"Yes, the graph G is bipartite.\" or \"No, the graph G is not bipartite.\""`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: \"Yes, the graph G is bipartite.\" or \"No, the graph G is not bipartite.\""`

---

### MINIMUM SPANNING TREE (MST)

#### Base Question:
```
"What is the minimum spanning tree of G?"
```

#### Output Formats:
- **Basic**: `" Output the result as a list."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."`

#### MST Algorithm (from mst.txt):
```
FUNCTION minimum_spanning_tree(edgelist):    
    INITIALIZE vertices with the unique vertices of edgelist
    # Initialize a mapping from each vertex to itself
    INITIALIZE vertex_to_set AS an empty dictionary
    FOR EACH vertex IN vertices:
        vertex_to_set[vertex] = vertex
    
    # Initialize a list to store edges of the MST
    INITIALIZE mst_edges AS an empty list
    # Iterate over edgelist to find MST using a simple union approach
    FOR EACH edge (vertex1, vertex2) IN edgelist:
        
    	IF vertex_to_set[vertex1] is not equal to vertex_to_set[vertex2]:
	    # Add edge to mst_edges
	    APPEND edge (vertex1, vertex2) TO mst_edges
	    PRINT mst_edges
	    PRINT vertex_to_set
            FOR EACH vertex IN vertices:
		# check if vertex in same set with vertex2
                IF vertex_to_set[vertex] is equal to vertex_to_set[vertex2]:
                    vertex_to_set[vertex] = vertex_to_set[vertex1]

    RETURN mst_edges
```

---

### TOPOLOGICAL SORTING

#### Base Question:
```
"What is the topological sorting of G?"
```

#### Output Formats:
- **Basic**: `" Output the result as a list."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."`

---

## Multi-Question Problems

These problems generate multiple questions per graph:

### NODE DEGREE

#### Base Question Pattern:
```
"What is the degree of node X?"
```
Where X is replaced with specific node numbers (typically 5 questions per graph).

#### Output Formats:
- **Basic**: `" Output the result like that: The degree of the node is ..."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The degree of the node is ..."`

### CONNECTED NODES

#### Base Question Pattern:
```
"Which nodes are the neighbors of node X?"
```

#### Output Formats:
- **Basic**: `" Output the result as a list."`
- **Algorithm**: `"Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."`

### CONNECTIVITY

#### Base Question Pattern:
```
"Is there a path between nodes X and Y?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'"`
- **Algorithm**: `"Follow the provided pseudocode step-by-step and show all steps. Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'"`

### SHORTEST PATH

#### Base Question Pattern:
```
"What is the shortest path length between nodes X and Y?"
```

#### Output Formats:
- **Basic**: `" Output the result like that: the shortest path length is..."`
- **Algorithm**: `" Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: the shortest path length is ..."`

---

## Simplification Methods (Experimental)

### `simplify_dif`
**Format**: `[Question stripped of "what is the number of nodes"] + " Keep only the first list and output how many numbers it has."`

### `simplify_more`  
**Format**: `[Question stripped of "what is the number of nodes"] + " Keep only the first list and count how many 0 and 1 it has."`

---

## Key Implementation Details

1. **Prompt Order**: For pseudocode methods, the pseudocode is **prepended** to the question
2. **Chain-of-Thought**: The "Let's think step by step." is **appended** to the question
3. **Multiple Questions**: For multi-question problems, each individual question gets the full prompt treatment
4. **Output Format**: Always added after the base question but before reasoning prompts (CoT/BAG)
5. **Algorithm Format**: Uses "Follow the provided pseudocode step-by-step and show all steps." instead of basic output format

This structure allows for systematic experimentation across all combinations of problems and prompting techniques.