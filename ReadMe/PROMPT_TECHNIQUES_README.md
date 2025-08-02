# Graph Reasoning LLM Experiments: Prompt Techniques & Problem Types

## Overview
This codebase implements a comprehensive framework for testing various prompt engineering techniques on graph reasoning problems with large language models. The system supports 10 different graph problem types using multiple prompting strategies to evaluate LLM performance on graph-based reasoning tasks.

## Prompt Techniques

### 1. **None/Baseline** (`none`)
- **Description**: Direct question without additional prompting techniques
- **Usage**: Baseline comparison for other methods
- **Example**: "What is the number of nodes of G?"

### 2. **Chain-of-Thought** (`cot`)
- **Description**: Encourages step-by-step reasoning
- **Prompt Addition**: "Let's think step by step."
- **Usage**: Appended to the end of questions to encourage explicit reasoning steps

### 3. **Build-a-Graph** (`bag`)
- **Description**: Instructs the model to explicitly construct the graph first
- **Prompt Addition**: "Let's construct a graph with the nodes and edges first."
- **Usage**: Helps models visualize the graph structure before solving

### 4. **Algorithm-Based** (`alg`)
- **Description**: Provides algorithmic pseudocode for the specific problem
- **Files**: Located in `pseudocodes/{problem}.txt`
- **Usage**: Prepends algorithm description to guide systematic problem solving
- **Example Instructions**: "Follow the provided pseudocode step-by-step and show all steps."

### 5. **One-Shot Examples** (`default_1_shot`)
- **Description**: Includes a worked example before the actual question
- **Files**: Located in `pseudocodes/{problem}_1_shot.txt`
- **Usage**: Provides context through example problem-solution pairs

### 6. **One-Shot with Pseudocode** (`1_shot_pseudo`)
- **Description**: Combines one-shot examples with algorithmic pseudocode
- **Files**: Located in `pseudocodes/{problem}_1_shot_pseudo.txt`
- **Usage**: Most comprehensive approach combining examples and algorithms

### 7. **Simplification Methods**
- **`simplify_dif`**: "Keep only the first list and output how many numbers it has."
- **`simplify_more`**: "Keep only the first list and count how many 0 and 1 it has."
- **Usage**: Experimental approaches for specific counting problems

## Graph Problem Types

### Single-Question Problems
These problems generate one question per graph instance:

#### 1. **Node Count** (`node_count`)
- **Task**: Count total number of nodes in the graph
- **Output Format**: "The number of nodes is ..."
- **Single Question**: Yes

#### 2. **Edge Count** (`edge_count`)
- **Task**: Count total number of edges in the graph
- **Output Format**: "The number of edges is ..."
- **Single Question**: Yes

#### 3. **Connected Components Count** (`connected_components_count`)
- **Task**: Count number of disconnected components
- **Output Format**: "The number of connected components is..."
- **Single Question**: Yes

#### 4. **Cycle Detection** (`cycle_check`)
- **Task**: Determine if graph contains cycles
- **Output Format**: "There is a cycle" or "There is no cycle."
- **Single Question**: Yes

#### 5. **Bipartite Check** (`bipartite`)
- **Task**: Determine if graph is bipartite
- **Output Format**: "Yes, the graph G is bipartite." or "No, the graph G is not bipartite."
- **Single Question**: Yes

#### 6. **Minimum Spanning Tree** (`mst`)
- **Task**: Find minimum spanning tree
- **Output Format**: List of edges
- **Single Question**: Yes

#### 7. **Topological Sorting** (`topological_sorting`)
- **Task**: Find valid topological ordering (DAGs only)
- **Output Format**: List of nodes in topological order
- **Single Question**: Yes

### Multi-Question Problems
These problems generate multiple questions per graph instance:

#### 8. **Node Degree** (`node_degree`)
- **Task**: Find degree of specific nodes
- **Output Format**: "The degree of the node is ..."
- **Multi-Question**: Yes (typically 5 questions per graph)
- **Question Pattern**: "What is the degree of node X?" for different nodes

#### 9. **Connected Nodes** (`connected_nodes`)
- **Task**: Find neighbors/adjacent nodes for specific nodes
- **Output Format**: List of connected nodes
- **Multi-Question**: Yes (typically 5 questions per graph)
- **Question Pattern**: "Which nodes are the neighbors of node X?"

#### 10. **Connectivity** (`connectivity`)
- **Task**: Check if two specific nodes are connected by a path
- **Output Format**: "Yes the two nodes are connected." or "No the two nodes are not connected."
- **Multi-Question**: Yes (typically 10 questions per graph)
- **Question Pattern**: "Is there a path between nodes X and Y?"

#### 11. **Shortest Path** (`shortest_path`)
- **Task**: Find shortest path length between two specific nodes
- **Output Format**: "The shortest path length is..."
- **Multi-Question**: Yes (typically 5 questions per graph)
- **Question Pattern**: "What is the shortest path length between nodes X and Y?"

#### 12. **Edge Existence** (`edge_existence`)
- **Task**: Check if direct edge exists between two nodes
- **Output Format**: Boolean response
- **Multi-Question**: Yes (typically 10 questions per graph)
- **Question Pattern**: "Is there an edge between nodes X and Y?"

## Graph Representations

The system supports two graph input formats:

### 1. **Edge List** (`edgelist`)
- **Format**: List of tuples representing edges
- **Example**: `[(0, 2), (1, 6), (2, 4), (3, 5), (4, 5)]`
- **Default**: Primary format used in experiments

### 2. **Adjacency Matrix** (`adj`)
- **Format**: 2D binary matrix
- **Example**: `[[0, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]]`
- **Usage**: Alternative representation for comparison

## Graph Types & Sizes

### Graph Types
- **Erdős–Rényi** (`er`): Random graphs with edge probability
- **Barabási–Albert** (`ba`): Scale-free networks (when implemented)
- **Path** (`path`): Linear chain graphs
- **Complete** (`complete`): Fully connected graphs
- **Stochastic Block Model** (`sbm`): Community structure graphs
- **Star** (`star`): Central hub with spokes
- **DAG** (`dag`): Directed Acyclic Graphs (for topological sorting)
- **Bipartite** (`bipartite`): Two-partition graphs

### Graph Sizes
- **Small**: ~7-15 nodes, manageable complexity
- **Medium**: ~20-50 nodes, moderate complexity  
- **Large**: ~50-100 nodes, high complexity

## Question Generation System

Questions are pre-generated and stored in JSON format:
```json
{
  "adj": {
    "node_count": "Let G be a graph. The adjacency matrix...",
    "node_degree": ["Question 1", "Question 2", ...]
  },
  "edgelist": {
    "node_count": "Let G be a graph. The edgelist...",
    "node_degree": ["Question 1", "Question 2", ...]
  }
}
```

### Multi-Question Strategy
For problems requiring multiple questions:
1. **Node-specific queries**: Target different nodes (degree, neighbors)
2. **Pair-wise queries**: Test different node pairs (connectivity, shortest path, edge existence)
3. **Systematic coverage**: Ensure diverse test cases per graph

## Experimental Design

### Model Configuration
- **Models**: GPT-3.5-turbo, GPT-4, Text-Davinci-003, Code-Davinci-002
- **Temperature**: 0 (deterministic)
- **Max Tokens**: 4000

### Evaluation Metrics
- **Accuracy**: Percentage of correct answers
- **Problem-specific validation**: Custom parsers for each problem type
- **Multi-question aggregation**: Per-graph and per-question accuracy

### File Structure
```
graphs/{type}/{size}/          # Graph files (.txt)
graphs_questions/{type}/{size}/ # Question files (.txt)  
pseudocodes/                   # Algorithm prompts
exp_results/                   # Detailed results
experiments/                   # Summary statistics
```

## Key Implementation Notes

1. **Answer Parsing**: Custom parsers for each problem type to extract structured answers from LLM responses

2. **Ground Truth Verification**: NetworkX-based reference implementations for validation

3. **Prompt Engineering**: Modular system allowing combination of techniques (e.g., "cot + pseudocode")

4. **Batch Processing**: Support for running comprehensive experiments across all combinations

5. **Error Handling**: Robust parsing with fallback strategies for malformed LLM responses

This framework provides a systematic approach to evaluating how different prompting strategies affect LLM performance on graph reasoning tasks, with particular attention to scalability and multi-question problem complexity.