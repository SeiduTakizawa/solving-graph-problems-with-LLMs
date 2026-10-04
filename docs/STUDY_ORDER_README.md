# Study Order Guide - Understanding the Refactored Code

## 📚 Recommended Reading Order

To understand the refactored graph reasoning experiments code, study the files in this specific order:

### 1️⃣ **START HERE: `experiment_config.py`** ⭐
**Why first:** Foundation of everything - defines data structures and paths

**What to focus on:**
- `ExperimentConfig` dataclass - understand what parameters control experiments
- `PathManager` class - see how file paths are organized
- Look at `__post_init__()` method to understand parameter normalization

**Key concepts:** Configuration management, file path organization

---

### 2️⃣ **`file_io.py`** 📁
**Why second:** Shows how data flows in/out of the system

**What to focus on:**
- `FileManager` static methods - basic file operations
- `GraphDataLoader` class - how graphs and questions are loaded
- `load_experiment_data()` method - the main data loading function

**Key concepts:** File handling, graph loading, JSON parsing

---

### 3️⃣ **`prompt_builder.py`** 💬
**Why third:** Core logic for how questions are formatted

**What to focus on:**
- `PromptBuilder.build_prompt()` - main entry point
- `_build_default_prompt()` vs `_build_algorithm_prompt()` - different methods
- `_add_output_formatting()` - how different problems get formatted

**Key concepts:** Prompt engineering, method-specific formatting

---

### 4️⃣ **`result_processor.py`** 🔍
**Why fourth:** How answers are validated and processed

**What to focus on:**
- `ResultProcessor.compute_ground_truth()` - how correct answers are calculated
- `parse_answer()` method - how model responses are parsed
- `ExperimentResultCollector` - how statistics are tracked

**Key concepts:** Answer parsing, validation, statistics collection

---

### 5️⃣ **`experiment_runner.py`** 🏃‍♂️
**Why fifth:** Orchestrates everything together

**What to focus on:**
- `SingleExperimentRunner.run_single_file()` - processes one graph
- `ExperimentRunner.run_experiment()` - main experiment loop
- How all the previous components work together

**Key concepts:** Pipeline orchestration, experiment flow

---

### 6️⃣ **`run_experiments.py`** 🚀
**Why last:** Simple entry point that ties everything together

**What to focus on:**
- `main()` function - command line argument parsing
- How `ExperimentConfig` is created from CLI args
- How `run_single_experiment()` is called

**Key concepts:** CLI interface, configuration creation

---

## 🎯 Study Strategy

### Phase 1: Core Concepts (Files 1-2)
```bash
# Read these to understand the data structures
1. experiment_config.py  
2. file_io.py
```

### Phase 2: Processing Logic (Files 3-4)
```bash
# Read these to understand the business logic
3. prompt_builder.py
4. result_processor.py
```

### Phase 3: Integration (Files 5-6)
```bash
# Read these to see how everything connects
5. experiment_runner.py
6. run_experiments.py
```

## 🔍 What to Look For in Each File

### `experiment_config.py`
- [ ] How `ExperimentConfig` dataclass is structured
- [ ] What each parameter (problem, method, size, etc.) controls
- [ ] How `PathManager` builds file paths
- [ ] The `get_all_experiment_configs()` function for batch runs

### `file_io.py`
- [ ] `FileManager.load_graph()` - NetworkX graph loading
- [ ] `FileManager.load_questions()` - JSON question parsing
- [ ] `GraphDataLoader.get_edgelist()` - graph format conversion
- [ ] Error handling for missing files

### `prompt_builder.py`
- [ ] Method dispatch in `build_prompt()`
- [ ] How different methods (cot, alg, pseudo) work
- [ ] Problem-specific output formatting
- [ ] Pseudocode file loading logic

### `result_processor.py`
- [ ] Ground truth computation for each problem type
- [ ] Answer parsing strategies (numbers, booleans, lists)
- [ ] Special cases for MST and topological sorting
- [ ] Statistics collection and summary formatting

### `experiment_runner.py`
- [ ] Single file processing pipeline
- [ ] How all components are initialized and used
- [ ] Result saving and statistics collection
- [ ] Progress tracking with tqdm

### `run_experiments.py`
- [ ] Argument parsing setup
- [ ] Configuration object creation
- [ ] Single vs batch experiment modes

## 💡 Quick Start Tips

**If you're in a hurry, focus on these key functions:**

1. `ExperimentConfig.__init__()` - understand the parameters
2. `PromptBuilder.build_prompt()` - see how prompts are created  
3. `ResultProcessor.compute_ground_truth()` - understand validation
4. `ExperimentRunner.run_experiment()` - see the main loop

**If you want to trace a complete example:**
- Start with `PIPELINE_README.md` to see the full execution flow
- Then study the files in the order above to understand each step

## 🚨 Don't Skip These Important Details

- **Configuration normalization** in `experiment_config.py` (bipartite/DAG handling)
- **Method dispatch logic** in `prompt_builder.py` (how different methods are handled)
- **Special answer parsing** in `result_processor.py` (MST, topological sorting)
- **Error handling** throughout all files

## 🎓 After Reading All Files

You should understand:
- ✅ How experiments are configured and paths are managed
- ✅ How graph data and questions are loaded from files
- ✅ How prompts are built for different problems and methods
- ✅ How model answers are parsed and validated
- ✅ How the complete experiment pipeline works
- ✅ How to run experiments and interpret results

**Next step:** Try modifying the code to add a new problem type or method!