"""
Configuration and path management for graph reasoning experiments.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    problem: str
    adj: str = "edgelist"
    model: str = "gpt-3.5-turbo"
    type: str = "er"
    size: str = "small"
    temperature: int = 0
    token: int = 4000
    method: str = "none"
    structured_output: bool = False
    
    def __post_init__(self):
        """Normalize problem and type settings."""
        if self.problem == "bipartite" or self.type == "bipartite":
            self.type = "bipartite"
            self.problem = "bipartite"
        
        if self.problem == "topological_sorting" or self.type == "dag":
            self.type = "dag"
            self.problem = "topological_sorting"


class PathManager:
    """Manages file paths for experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    @property
    def graphs_dir(self) -> str:
        """Directory containing graph files."""
        return os.path.join('graphs', self.config.type, self.config.size)
    
    @property
    def questions_dir(self) -> str:
        """Directory containing question files."""
        return os.path.join('graphs_questions', self.config.type, self.config.size)
    
    @property
    def results_dir(self) -> str:
        """Directory for storing results."""
        return os.path.join(
            'exp_results', self.config.problem, self.config.adj, 
            self.config.type, self.config.size, self.config.model, self.config.method
        )
    
    @property
    def experiment_dir(self) -> str:
        """Directory for experiment summaries."""
        return os.path.join(
            'experiments', self.config.problem, self.config.adj,
            self.config.type, self.config.size, self.config.model
        )
    
    def get_graph_file(self, filename: str) -> str:
        """Get full path to a graph file."""
        return os.path.join(self.graphs_dir, filename)
    
    def get_question_file(self, filename: str) -> str:
        """Get full path to a question file."""
        return os.path.join(self.questions_dir, filename)
    
    def get_result_file(self, filename: str) -> str:
        """Get full path to a result file."""
        return os.path.join(self.results_dir, filename)
    
    def get_experiment_summary_file(self) -> str:
        """Get full path to experiment summary file."""
        return os.path.join(self.experiment_dir, f'{self.config.method}.txt')
    
    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.results_dir, self.experiment_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)


def get_all_experiment_configs() -> List[ExperimentConfig]:
    """Generate all experiment configurations for batch runs."""
    problems = [
        "node_count", "edge_count", "node_degree", "connected_nodes", 
        "connected_components_count", "cycle_check", "shortest_path", 
        "minimum_spanning_tree", "topological_sorting", "bipartite"
    ]
    
    sizes = ["small", "medium", "large"]
    types = ["er"]  # Can be expanded: ["er", "ba", "path", "complete", "sbm", "sfn", "star"]
    methods = ["none", "default_1_shot", "cot", "alg", "1_shot_pseudo"]
    model = 'gpt-3.5-turbo'
    
    configs = []
    for problem in problems:
        for size in sizes:
            for graph_type in types:
                for method in methods:
                    config = ExperimentConfig(
                        problem=problem, size=size, type=graph_type, 
                        method=method, model=model
                    )
                    configs.append(config)
    
    return configs