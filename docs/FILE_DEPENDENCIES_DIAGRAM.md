# File Dependencies and Call Graph

## 📊 Visual Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN ENTRY POINT                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  run_experiments.py                          │ │
│  │                          [CLI Interface]                               │ │
│  └─────────────────────────────┬───────────────────────────────────────────┘ │
└──────────────────────────────────┼─────────────────────────────────────────────┘
                                 │
                                 ▼ imports & calls
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     experiment_runner.py                               │ │
│  │              [ExperimentRunner, SingleExperimentRunner]                │ │
│  └─────┬───────────────┬─────────────────┬─────────────────┬───────────────┘ │
└────────┼───────────────┼─────────────────┼─────────────────┼─────────────────┘
         │               │                 │                 │
         ▼ imports       ▼ imports         ▼ imports         ▼ imports
┌────────────────┐ ┌──────────────┐ ┌───────────────┐ ┌─────────────────┐
│CONFIGURATION   │ │ FILE I/O     │ │ PROMPT LOGIC  │ │ RESULT LOGIC    │
│                │ │              │ │               │ │                 │
│experiment_     │ │ file_io.py   │ │ prompt_       │ │ result_         │
│config.py       │ │              │ │ builder.py    │ │ processor.py    │
│                │ │ [FileManager]│ │               │ │                 │
│[ExperimentConfig│ │[GraphData    │ │[PromptBuilder]│ │[ResultProcessor]│
│ PathManager]   │ │ Loader]      │ │               │ │[ResultCollector]│
└────────────────┘ └──────────────┘ └───────────────┘ └─────────────────┘
         ▲                │                 │                 │
         │                │                 │                 │
         └────────────────┼─────────────────┼─────────────────┘
                          │                 │
                          ▼ imports         ▼ imports
                 ┌─────────────────┐ ┌──────────────────┐
                 │ EXTERNAL DEPS   │ │ CORE FUNCTIONS   │
                 │                 │ │                  │
                 │ networkx        │ │ graph_algorithms.py │
                 │ json            │ │ utils.py         │
                 │ os              │ │                  │
                 │ tqdm            │ │ [Ground truth    │
                 │                 │ │  functions,      │
                 │                 │ │  API calls,      │
                 │                 │ │  Answer parsing] │
                 └─────────────────┘ └──────────────────┘
```

## 🔄 Call Flow During Execution

```
┌─ run_experiments.py ─┐
│ main()                          │
│  ├─ ArgumentParser()            │
│  ├─ ExperimentConfig() ─────────┼─── experiment_config.py
│  └─ run_single_experiment() ────┼─── experiment_runner.py
└─────────────────────────────────┘     │
                                        │
┌─ experiment_runner.py ──────────────────┘
│ ExperimentRunner()               │
│  ├─ PathManager() ──────────────┼─── experiment_config.py  
│  ├─ SingleExperimentRunner()    │
│  ├─ ExperimentResultCollector() ┼─── result_processor.py
│  └─ run_experiment()            │
│      ├─ GraphDataLoader() ──────┼─── file_io.py
│      ├─ run_single_file() ──────┼─── [LOOP for each graph]
│      │   ├─ load_experiment_data()┼─── file_io.py
│      │   ├─ build_prompt() ─────┼─── prompt_builder.py
│      │   ├─ get_openai_response()┼─── utils.py
│      │   ├─ compute_ground_truth()┼─── result_processor.py → graph_algorithms.py
│      │   └─ parse_answer() ─────┼─── result_processor.py → utils.py
│      ├─ save_result_file() ─────┼─── file_io.py
│      └─ save_experiment_summary()┼─── file_io.py
└─────────────────────────────────┘
```

## 📋 Import Dependencies Matrix

| File | Imports From | Provides To |
|------|-------------|-------------|
| `run_experiments.py` | `experiment_config`, `experiment_runner` | *Entry Point* |
| `experiment_runner.py` | `experiment_config`, `prompt_builder`, `result_processor`, `file_io`, `utils` | `run_experiments` |
| `experiment_config.py` | `os`, `dataclasses`, `typing` | `experiment_runner`, `prompt_builder`, `result_processor` |
| `prompt_builder.py` | `experiment_config`, `typing` | `experiment_runner` |
| `result_processor.py` | `graph_algorithms`, `utils`, `typing` | `experiment_runner` |
| `file_io.py` | `os`, `json`, `networkx`, `typing` | `experiment_runner` |

## 🎯 Key Relationships

### **Central Hub: `experiment_runner.py`**
- **Imports from:** All other modules
- **Coordinates:** The entire experiment pipeline
- **Role:** Main orchestrator

### **Foundation: `experiment_config.py`**
- **Imported by:** `experiment_runner`, `prompt_builder`, `result_processor`
- **Provides:** Configuration and path management
- **Role:** Foundation layer

### **Utilities: `utils.py` & `graph_algorithms.py`**
- **Imported by:** `result_processor`, `experiment_runner`
- **Provides:** Core algorithms and API calls
- **Role:** Implementation details

### **Specialized Workers:**
- **`prompt_builder.py`:** Only used by `experiment_runner`
- **`result_processor.py`:** Only used by `experiment_runner`  
- **`file_io.py`:** Only used by `experiment_runner`

## 🔀 Data Flow

```
CLI Args → ExperimentConfig → PathManager → GraphDataLoader → Graph Data
                ↓                           ↓                    ↓
         PromptBuilder ← Question Data ← FileManager ← Question Files
                ↓
         Formatted Prompt → OpenAI API → Raw Answer
                                            ↓
ResultProcessor ← Ground Truth ← graph_algorithms.py
       ↓              ↑
Parsed Answer → Validation → Statistics → Summary Files
```

## 🏗️ Architecture Pattern

This follows a **"Hub and Spoke"** architecture:
- **Hub:** `experiment_runner.py` (coordinates everything)
- **Spokes:** Specialized modules for specific tasks
- **Foundation:** `experiment_config.py` (used by multiple spokes)
- **Utilities:** `utils.py` & `graph_algorithms.py` (support functions)

## 💡 Design Benefits

1. **Clear separation:** Each file has one responsibility
2. **Minimal coupling:** Most files only talk to the hub
3. **Easy testing:** Each module can be tested independently
4. **Maintainable:** Changes are localized to specific files
5. **Extensible:** New functionality can be added as new spokes