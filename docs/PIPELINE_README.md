# Graph Reasoning Pipeline - Execution Flow

## Example: Node Count Problem with Small Graphs and GPT-3.5-turbo

This document traces the complete execution pipeline when running:
```bash
python run_experiments.py --problem node_count --method none --size small --model gpt-3.5-turbo
```

## 📊 Pipeline Overview

```
CLI Input → Config Creation → Experiment Runner → File Processing → Model Query → Result Processing → Output
```

## 🔄 Detailed Execution Flow

### 1. **Entry Point** - `run_experiments.py`

```python
# Command line execution starts here
def main():
    parser = argparse.ArgumentParser(description="Graph reasoning experiments")
    args = parser.parse_args()  # Parses --problem node_count --method none --size small
    
    # Creates configuration object
    config = ExperimentConfig(
        problem="node_count",     # From --problem
        method="none",           # From --method  
        size="small",           # From --size
        model="gpt-3.5-turbo"   # From --model (or default)
    )
    
    # Calls the main experiment runner
    run_single_experiment(config) ✓
```

### 2. **Experiment Setup** - `experiment_runner.py`

```python
def run_single_experiment(config: ExperimentConfig) -> float:
    # Creates the main experiment coordinator
    runner = ExperimentRunner(config) ✓
    return runner.run_experiment() ✓

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.path_manager = PathManager(config) ✓          # Manages file paths
        self.single_runner = SingleExperimentRunner(config) ✓  # Handles individual files
        self.result_collector = ExperimentResultCollector() ✓  # Collects statistics
```

### 3. **Path Management** - `experiment_config.py`

```python
class PathManager:
    def __init__(self, config):
        # For our example, creates these paths:
        self.graphs_dir = "../graphs/er/small"  # Graph files location
        self.questions_dir = "../graphs_questions/er/small"  # Question files location
        self.results_dir = "results/exp_results/node_count/edgelist/er/small/gpt-3.5-turbo/none"
        self.experiment_dir = "../experiments/node_count/edgelist/er/small/gpt-3.5-turbo"

    def ensure_directories_exist(self):
# Creates result directories if they don't exist ✓
```

### 4. **Main Experiment Loop** - `experiment_runner.py`

```python
def run_experiment(self) -> float:
    print(self.config)  # Prints experiment configuration
    
    # Setup directories
    self.path_manager.ensure_directories_exist() ✓
    
    # Initialize file loader
    graph_loader = GraphDataLoader(
        graphs_dir="graphs/er/small",
        questions_dir="graphs_questions/er/small"
    ) ✓
    
    # Get list of graph files to process
    files = graph_loader.get_available_files()  # Gets all .txt files in graphs/er/small/
    numbered_files = ["0.txt", "1.txt", "2.txt", ...]  # Converts to numbered format
    
    # Process each file
    for i in tqdm(range(len(numbered_files))):
        filename = numbered_files[i]  # e.g., "0.txt"
        
        # Process single file ✓
        result_info = self.single_runner.run_single_file(filename, graph_loader) ✓
        
        # Save and collect results ✓
        self._save_result_file(result_info) ✓
        self.result_collector.add_result(filename, result_info['is_correct']) ✓
```

### 5. **Single File Processing** - `experiment_runner.py`

```python
class SingleExperimentRunner:
    def run_single_file(self, filename: str, graph_loader: GraphDataLoader):
        # Load graph and questions for "0.txt"
        graph, questions = graph_loader.load_experiment_data(
            filename="0.txt", 
            problem="node_count"
        ) ✓
        
        # Build the prompt
        formatted_questions = self.prompt_builder.build_prompt(questions) ✓
        
        # Convert graph to edgelist format
        edgelist = graph_loader.get_edgelist(graph) ✓  # [(1,2), (2,3), (1,3)]
        
        # Query the language model
        answers = self._query_model(formatted_questions) ✓
        
        # Process and validate results
        return self._process_results(filename, questions, formatted_questions, answers, edgelist) ✓
```

### 6. **File Loading** - `file_io.py`

```python
class GraphDataLoader:
    def load_experiment_data(self, filename: str, problem: str):
        # Load graph from "graphs/er/small/0.txt"
        graph_path = "graphs/er/small/0.txt"
        graph = FileManager.load_graph(graph_path, is_directed=False) ✓
        # Returns NetworkX graph object
        
        # Load questions from "graphs_questions/er/small/0.txt"  
        question_path = "graphs_questions/er/small/0.txt"
        questions = FileManager.load_questions(question_path, "node_count") ✓
        # Returns: "What is the number of nodes of G?"
        
        return graph, questions

    def get_edgelist(self, graph):
        # Converts NetworkX graph to list of tuples
        return [(int(u), int(v)) for u, v in graph.edges()] ✓
        # Returns: [(1, 2), (2, 3), (1, 3)]
```

### 7. **Prompt Building** - `prompt_builder.py`

```python
class PromptBuilder:
    def build_prompt(self, question: str) -> str:
        # For method="none", problem="node_count"
        
        # Since method is "none", goes to default prompt building
        question = self._build_default_prompt(question) ✓
        
    def _build_default_prompt(self, question: str) -> str:
        # No pseudocode for method="none"
        
        # Add output formatting for node_count
        question = self._add_output_formatting(question) ✓
        # "What is the number of nodes of G?" 
        # becomes:
        # "What is the number of nodes of G? Output the result like that: The number of nodes is ..."
        
        return question
```

### 8. **Model Querying** - `experiment_runner.py` → `utils.py`

```python
def _query_model(self, questions: str) -> str:
    # Calls OpenAI API
    return get_openai_response(questions, self.config) ✓

# In utils.py
@retry(wait=wait_random_exponential(min=1, max=100), stop=stop_after_attempt(500))
def get_openai_response(text, args):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Our specified model
        messages=[
            {"role": "system", "content": "You are a helpful graph problem assistant."},
            {"role": "user", "content": "What is the number of nodes of G? Output the result like that: The number of nodes is ..."}
        ],
        temperature=0
    )
    return completion.choices[0].message.content ✓
    # Returns: "Looking at the graph G, I can count the nodes. The number of nodes is 3."
```

### 9. **Result Processing** - `result_processor.py`

```python
def _process_results(self, filename, original_questions, formatted_questions, answers, edgelist):
    # Compute ground truth
    ground_truth = self.result_processor.compute_ground_truth(edgelist) ✓
    
class ResultProcessor:
    def compute_ground_truth(self, edgelist: List[Tuple[int, int]]):
        # For problem="node_count"
        return node_count(edgelist) ✓  # Calls function from graph_algorithms.py
        # For edgelist [(1,2), (2,3), (1,3)] returns: 3
    
    def parse_answer(self, answer: str, edgelist):
        # For problem="node_count"  
        return get_number_answer(answer) ✓  # Calls function from utils.py
        # Parses "The number of nodes is 3." → returns: 3
```

### 10. **Answer Parsing** - `utils.py`

```python
def get_number_answer(answer):
    # "Looking at the graph G, I can count the nodes. The number of nodes is 3."
    found_answer = find_number(answer) ✓
    return found_answer

def find_number(answer):
    answer_list = answer.split()[-10:]  # Gets last 10 words: ["nodes", "is", "3."]
    
    for i in range(1, len(answer_list)):
        word = answer_list[-i]  # Starts with "3."
        word = word.strip().strip("'").strip('"').strip(".").strip("*")  # Cleans to "3"
        try:
            answer_number = int(word)  # Converts to integer 3
            return answer_number ✓
        except ValueError:
            continue
    
    return -2  # If no number found
```

### 11. **Result Validation & Saving** - `experiment_runner.py`

```python
def _validate_and_format_results(self, filename, original_questions, formatted_questions, 
                                answers, parsed_answers, ground_truth):
    # Compare parsed answer (3) with ground truth (3)
    is_correct = (parsed_answers == ground_truth) ✓  # True
    
    return {
        'filename': "0.txt",
        'is_correct': True,
        'correct': "Correct",
        'ground_truth': 3,
        'found_answer': 3,
        'question': "What is the number of nodes of G? Output the result like that: The number of nodes is ...",
        'answer': "Looking at the graph G, I can count the nodes. The number of nodes is 3.",
        'multi_question': False
    }

def _save_result_file(self, result_info):
    # Saves to: results/exp_results/node_count/edgelist/er/small/gpt-3.5-turbo/none/0.txt
    FileManager.save_single_result(
        file_path="exp_results/node_count/edgelist/er/small/gpt-3.5-turbo/none/0.txt",
        correct="Correct",
        ground_truth=3,
        found_answer=3,
        question="What is the number of nodes of G? Output the result like that: The number of nodes is ...",
        answer="Looking at the graph G, I can count the nodes. The number of nodes is 3."
    ) ✓
```

### 12. **Statistics Collection** - `result_processor.py`

```python
class ExperimentResultCollector:
    def add_result(self, filename: str, is_correct: bool):
        self.total_count += 1      # Increments total
        if is_correct:
            self.correct_count += 1  # Increments correct count
            print("Correct")         # Prints success message
        else:
            self.not_solved.append(filename)  # Adds to failed list
```

### 13. **Final Summary** - `experiment_runner.py`

```python
def run_experiment(self):
    # ... after processing all files ...
    
    # Save experiment summary
    self._save_experiment_summary() ✓
    
    # Print final results
    summary = self.result_collector.get_summary()
    print(self.config)
    print(f"Solved correctly: {summary['solved_correctly']}")      # e.g., "Solved correctly: 85"
    print(f"Percentage solved correctly (%): {summary['percentage']}")  # e.g., "Percentage solved correctly (%): 0.85"
    
    return summary['percentage']

def _save_experiment_summary(self):
    # Creates summary text
    summary_text = self.result_collector.format_summary_text(
        method="none", 
        problem="node_count"
    )
    
    # Saves to: experiments/node_count/edgelist/er/small/gpt-3.5-turbo/none.txt
    FileManager.save_experiment_summary(
        file_path="experiments/node_count/edgelist/er/small/gpt-3.5-turbo/none.txt",
        summary_text="""none
node_count
Solved correctly: 85
Percentage solved correctly (%): 0.85
Total: 100

Not solved:
15.txt
23.txt
..."""
    ) ✓
```

## 📁 File Structure Created

After running the experiment, you'll see:

```
results/exp_results/node_count/edgelist/er/small/gpt-3.5-turbo/none/
├── 0.txt     # Detailed result for graph 0
├── 1.txt     # Detailed result for graph 1
├── 2.txt     # Detailed result for graph 2
└── ...

experiments/node_count/edgelist/er/small/gpt-3.5-turbo/
└── none.txt  # Summary statistics for the experiment
```

## 🎯 Key Function Call Chain

```
main() 
  → run_single_experiment(config)
    → ExperimentRunner(config).run_experiment()
      → SingleExperimentRunner.run_single_file() [for each graph file]
        → GraphDataLoader.load_experiment_data()
          → FileManager.load_graph() & FileManager.load_questions()
        → PromptBuilder.build_prompt()
        → get_openai_response() [API call]
        → ResultProcessor.compute_ground_truth() & parse_answer()
        → FileManager.save_single_result()
        → ExperimentResultCollector.add_result()
      → FileManager.save_experiment_summary()
```

## 💡 Summary

The pipeline efficiently processes each graph file through a clean sequence of:
1. **Load** graph and questions
2. **Build** formatted prompt  
3. **Query** GPT-3.5-turbo
4. **Parse** the response
5. **Validate** against ground truth
6. **Save** detailed results
7. **Collect** statistics
8. **Generate** final summary

Each step is handled by a focused module, making the entire pipeline maintainable and easy to debug!