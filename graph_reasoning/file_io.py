
"""
File I/O utilities for graph reasoning experiments.
"""
import os
import json
import networkx as nx
from typing import Dict, List, Union, Tuple


class FileManager:
    """Handles file operations for experiments."""
    
    @staticmethod
    def load_graph(file_path: str, is_directed: bool = False) -> nx.Graph:
        """Load a graph from adjacency list file."""
        if is_directed:
            return nx.read_adjlist(file_path, comments='#', create_using=nx.DiGraph)
        else:
            return nx.read_adjlist(file_path, comments='#')
    
    @staticmethod
    def load_questions(file_path: str, problem: str, adj_type: str = "edgelist") -> Union[str, List[str]]:
        """Load questions from JSON file."""
        with open(file_path, 'r') as f:
            question_str = f.read()
        
        question_json = json.loads(question_str)
        
        if problem in question_json.get(adj_type, {}):
            return question_json[adj_type][problem]
        else:
            return None
    
    @staticmethod
    def save_detailed_result(file_path: str, results: List[Dict]):
        """Save detailed results for multi-question problems."""
        with open(file_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"{result['correct']}\n\n")
                f.write(f"Ground truth: {result['ground_truth']}\n")
                f.write(f"Found answer: {result['found_answer']}\n\n")
                f.write(f"Question: {result['question']}\n\n")
            
            if results:
                f.write(f"Answer: {results[-1]['answer']}\n\n--------------------\n\n")
    
    @staticmethod
    def save_single_result(file_path: str, correct: str, ground_truth, found_answer, 
                          question: str, answer: str):
        """Save result for single-question problems."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{correct}\n\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Found answer: {found_answer}\n\n")
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}")
    
    @staticmethod
    def save_experiment_summary(file_path: str, summary_text: str):
        """Save experiment summary to file."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    
    @staticmethod
    def get_file_list(directory: str) -> List[str]:
        """Get list of files in directory."""
        if not os.path.exists(directory):
            return []
        return os.listdir(directory)
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if file exists."""
        return os.path.isfile(file_path)
    
    @staticmethod
    def save_json_result(file_path: str, json_data):
        """Save structured JSON result to file."""
        # Change extension to .json
        json_file_path = os.path.splitext(file_path)[0] + '.json'
        
        directory = os.path.dirname(json_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)


class GraphDataLoader:
    """Loads and processes graph data for experiments."""
    
    def __init__(self, graphs_dir: str, questions_dir: str):
        self.graphs_dir = graphs_dir
        self.questions_dir = questions_dir
    
    def load_experiment_data(self, filename: str, problem: str, 
                           is_directed: bool = False) -> Tuple[nx.Graph, Union[str, List[str]]]:
        """Load both graph and questions for a single experiment."""
        # Load graph
        graph_path = os.path.join(self.graphs_dir, filename)
        graph = FileManager.load_graph(graph_path, is_directed)
        
        # Load questions
        question_path = os.path.join(self.questions_dir, filename)
        questions = FileManager.load_questions(question_path, problem)
        
        return graph, questions
    
    def get_available_files(self) -> List[str]:
        """Get list of available experiment files."""
        return FileManager.get_file_list(self.graphs_dir)
    
    def get_edgelist(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """Convert graph to edgelist format."""
        return [(int(u), int(v)) for u, v in graph.edges()]