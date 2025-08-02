"""
Prompt building logic for different graph problems and methods.
"""
from typing import Union, List
from experiment_config import ExperimentConfig


class PromptBuilder:
    """Builds prompts for different graph problems and methods."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def build_prompt(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Build the complete prompt based on configuration."""
        if self.config.method in ["none", "cot", "bag"] or 'default' in self.config.method:
            return self._build_default_prompt(question)
        elif self.config.method in ["simplify_dif", "simplify_more"]:
            return self._build_simplify_prompt(question)
        elif self.config.method == "alg" or 'pseudo' in self.config.method:
            return self._build_algorithm_prompt(question)
        else:
            return question
    
    def _build_default_prompt(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Build prompt for default/cot/bag methods."""
        # Add default pseudocode if specified
        if 'default' in self.config.method:
            question = self._add_default_pseudocode(question)
        
        # Add problem-specific output formatting
        question = self._add_output_formatting(question)
        
        # Add chain-of-thought or bag prompts
        if self.config.method in ["cot", "bag"]:
            question = self._add_reasoning_prompt(question)
        
        return question
    
    def _build_simplify_prompt(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Build prompt for simplification methods."""
        if isinstance(question, list):
            return question  # Simplify methods only work with single questions
        
        question = question.strip("what is the number of nodes")
        
        if self.config.method == "simplify_dif":
            prompt = " Keep only the first list and output how many numbers it has."
        elif self.config.method == "simplify_more":
            prompt = " Keep only the first list and count how many 0 and 1 it has."
        else:
            prompt = ""
        
        return question + prompt
    
    def _build_algorithm_prompt(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Build prompt for algorithm/pseudocode methods."""
        # Load pseudocode
        pseudocode = self._load_pseudocode()
        
        # Add algorithm-specific output formatting
        question = self._add_algorithm_output_formatting(question)
        
        # Prepend pseudocode to question
        if isinstance(question, list):
            for i in range(len(question)):
                question[i] = pseudocode + " " + question[i]
        else:
            question = pseudocode + " " + question
        
        return question
    
    def _add_default_pseudocode(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Add default pseudocode to question."""
        try:
            name = self.config.method.replace("default", "")
            with open(f"pseudocodes/{self.config.problem}{name}.txt", "r") as f:
                prompt = f.read()
            
            if isinstance(question, list):
                for i in range(len(question)):
                    question[i] = prompt + " " + question[i]
            else:
                question = prompt + " " + question
        except FileNotFoundError:
            pass  # If pseudocode file doesn't exist, continue without it
        
        return question
    
    def _load_pseudocode(self) -> str:
        """Load pseudocode for algorithm methods."""
        try:
            if self.config.method == "alg":
                with open(f"pseudocodes/{self.config.problem}.txt", "r") as f:
                    return f.read()
            elif 'pseudo' in self.config.method:
                name = self.config.method.replace('pseudo', '')
                with open(f"pseudocodes/{self.config.problem}_{name}pseudo.txt", "r", encoding="utf-8") as f:
                    return f.read()
        except FileNotFoundError:
            pass
        
        return ""
    
    def _add_output_formatting(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Add problem-specific output formatting."""
        formatters = {
            "node_count": " Output the result like that: The number of nodes is ...",
            "edge_count": " Output the result like that: The number of edges is ...",
            "connected_components_count": " Output the result like that: The number of connected components is...",
            "mst": " Output the result as a list.",
            "topological_sorting": " Output the result as a list.",
            "bipartite": ' Output the result like that: "Yes, the graph G is bipartite." or "No, the graph G is not bipartite."',
            "cycle_check": ' Output the result like that: "There is a cycle" or "There is no cycle."'
        }
        
        list_formatters = {
            "node_degree": " Output the result like that: The degree of the node is ...",
            "connected_nodes": " Output the result as a list.",
            "shortest_path": " Output the result like that: the shortest path length is...",
            "connectivity": " Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'"
        }
        
        if self.config.problem in formatters:
            if isinstance(question, str):
                question += formatters[self.config.problem]
        
        if self.config.problem in list_formatters and isinstance(question, list):
            formatter = list_formatters[self.config.problem]
            for i in range(len(question)):
                question[i] += formatter
        
        return question
    
    def _add_algorithm_output_formatting(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Add algorithm-specific output formatting."""
        formatters = {
            "node_count": " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of nodes is ...",
            "edge_count": " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of edges is ...",
            "cycle_check": ' Follow the provided pseudocode step-by-step and show all steps. Output the result like that: "There is a cycle" or "There is no cycle."',
            "connected_components_count": ' Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The number of connected components is ...',
            "mst": " Follow the provided pseudocode step-by-step and show all steps. Output the result as a list.",
            "topological_sorting": " Follow the provided pseudocode step-by-step and show all steps. Output the result as a list.",
            "bipartite": ' Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: "Yes, the graph G is bipartite." or "No, the graph G is not bipartite."'
        }
        
        list_formatters = {
            "node_degree": " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The degree of the node is ...",
            "shortest_path": " Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: the shortest path length is ...",
            "connectivity": "Follow the provided pseudocode step-by-step and show all steps. Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'",
            "connected_nodes": "Follow the provided pseudocode step-by-step and show all steps. Output the result as a list."
        }
        
        if self.config.problem in formatters:
            if isinstance(question, str):
                question += formatters[self.config.problem]
        
        if self.config.problem in list_formatters and isinstance(question, list):
            formatter = list_formatters[self.config.problem]
            for i in range(len(question)):
                question[i] += " " + formatter
        
        return question
    
    def _add_reasoning_prompt(self, question: Union[str, List[str]]) -> Union[str, List[str]]:
        """Add chain-of-thought or bag reasoning prompts."""
        if self.config.method == "cot":
            cot_prompt = "Let's think step by step."
        elif self.config.method == "bag":
            cot_prompt = "Let's construct a graph with the nodes and edges first."
        else:
            return question
        
        if isinstance(question, list):
            for i in range(len(question)):
                question[i] = question[i] + " " + cot_prompt
        else:
            question = question + " " + cot_prompt
        
        return question