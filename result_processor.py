"""
Result processing and validation for graph reasoning experiments.
"""
import re
from typing import Union, List, Tuple, Any
from new_functions import *
from utils import *


class ResultProcessor:
    """Processes and validates experiment results."""
    
    def __init__(self, problem: str):
        self.problem = problem
    
    def compute_ground_truth(self, edgelist: List[Tuple[int, int]], 
                           questions: Union[str, List[str]] = None, 
                           **kwargs) -> Union[Any, List[Any]]:
        """Compute ground truth solution for the problem."""
        if self.problem == "node_count":
            return node_count(edgelist)
        elif self.problem == "edge_count":
            return edge_count(edgelist)
        elif self.problem == "connected_components_count":
            return connected_components_count(edgelist)
        elif self.problem == "cycle_check":
            return cycle_check(edgelist)
        elif self.problem == "bipartite":
            return is_bipartite(edgelist)
        elif self.problem in ["node_degree", "connected_nodes", "connectivity", "shortest_path"]:
            return self._compute_multi_question_ground_truth(edgelist, questions)
        elif self.problem == "mst":
            # For MST, ground truth is computed during answer parsing
            return None
        elif self.problem == "topological_sorting":
            # For topological sorting, ground truth is computed during answer parsing
            return None
        else:
            raise ValueError(f"Unknown problem type: {self.problem}")
    
    def _compute_multi_question_ground_truth(self, edgelist: List[Tuple[int, int]], 
                                           questions: List[str]) -> List[Any]:
        """Compute ground truth for problems with multiple questions."""
        ground_truths = []
        
        for question in questions:
            if self.problem == "node_degree":
                node = self._extract_node_from_question(question)
                ground_truths.append(node_degree(edgelist, node))
            elif self.problem == "connected_nodes":
                node = self._extract_node_from_question(question)
                ground_truths.append(connected_nodes(edgelist, node))
            elif self.problem == "connectivity":
                source, target = self._extract_two_nodes_from_question(question)
                ground_truths.append(connectivity(edgelist, source, target))
            elif self.problem == "shortest_path":
                source, target = self._extract_two_nodes_from_question(question)
                ground_truths.append(shortest_path(edgelist, source, target))
        
        return ground_truths
    
    def parse_answer(self, answer: Union[str, List[str]], 
                    edgelist: List[Tuple[int, int]]) -> Union[Any, List[Any]]:
        """Parse the model's answer into the expected format."""
        if isinstance(answer, list):
            return [self._parse_single_answer(ans, edgelist) for ans in answer]
        else:
            return self._parse_single_answer(answer, edgelist)
    
    def _parse_single_answer(self, answer: str, edgelist: List[Tuple[int, int]]) -> Any:
        """Parse a single answer string."""
        if self.problem in ["node_count", "edge_count", "connected_components_count", "node_degree", "shortest_path"]:
            return get_number_answer(answer)
        elif self.problem == "cycle_check":
            return cycle_get_boolean_answer(answer)
        elif self.problem == "bipartite":
            return bipartite_get_boolean_answer(answer)
        elif self.problem == "connectivity":
            return connectivity_get_boolean_answer(answer)
        elif self.problem == "connected_nodes":
            # For connected_nodes, we need the node info which should be extracted elsewhere
            return self._parse_connected_nodes_answer(answer, edgelist)
        elif self.problem == "mst":
            parsed_answer, ground_truth = mst_get_answer(answer, edgelist)
            return parsed_answer
        elif self.problem == "topological_sorting":
            parsed_answer, ground_truth = topological_get_answer(answer, edgelist)
            return parsed_answer
        else:
            return answer
    
    def _parse_connected_nodes_answer(self, answer: str, edgelist: List[Tuple[int, int]]) -> Any:
        """Parse connected nodes answer - needs special handling."""
        # This is a simplified version - in full implementation would need node parameter
        counter = 1
        found_answer = answer.split("\n")[-counter]
        while "[" not in found_answer and "]" not in found_answer:
            counter += 1
            if len(answer.split("\n")) >= counter:
                found_answer = answer.split("\n")[-counter]
            if counter == 10:
                break
        
        if "[" in found_answer and "]" in found_answer:
            found_answer = found_answer[found_answer.index("["):found_answer.index("]")+1].strip(".").strip()
            try:
                return eval(found_answer)
            except:
                pass
        
        return found_answer
    
    def _extract_node_from_question(self, question: str) -> int:
        """Extract node number from question string."""
        # Remove common prompt suffixes
        for suffix in [
            " Output the result like that: The degree of the node is ...",
            " Output the result as a list.",
            " Follow the provided pseudocode step-by-step and show all steps. Output the result like that: The degree of the node is ...",
            " Follow the provided pseudocode step-by-step and show all steps. Output the result as a list.",
            "Let's think step by step.",
            "Let's construct a graph with the nodes and edges first."
        ]:
            question = question.replace(suffix, "")
        
        # Extract the last number from the question
        node = int(question.split()[-1].strip('"').strip("?").strip())
        return node
    
    def _extract_two_nodes_from_question(self, question: str) -> Tuple[int, int]:
        """Extract two node numbers from question string."""
        # Remove common prompt suffixes
        for suffix in [
            " Output the result like that: the shortest path length is...",
            " Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'",
            " Follow the provided pseudocode step-by-step and show all the steps. Output the result like that: the shortest path length is ...",
            "Follow the provided pseudocode step-by-step and show all steps. Output the result like that: 'Yes the two nodes are connected.' or 'No the two nodes are not connected.'",
            "Let's think step by step.",
            "Let's construct a graph with the nodes and edges first."
        ]:
            question = question.replace(suffix, "")
        
        question = question.strip()
        
        # Extract the last two numbers from the question
        target_node = int(question.split()[-1].strip('"').strip("?").strip())
        source_node = int(question.split()[-3].strip('"').strip("?").strip())
        return source_node, target_node
    
    def validate_answer(self, parsed_answer: Any, ground_truth: Any) -> bool:
        """Validate if the parsed answer matches ground truth."""
        if self.problem == "connected_nodes":
            # For connected nodes, compare as sets
            if isinstance(parsed_answer, list) and isinstance(ground_truth, list):
                return set(parsed_answer) == set(ground_truth)
        
        return parsed_answer == ground_truth


class ExperimentResultCollector:
    """Collects and summarizes experiment results."""
    
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0
        self.not_solved = []
    
    def add_result(self, filename: str, is_correct: bool):
        """Add a single result."""
        self.total_count += 1
        if is_correct:
            self.correct_count += 1
        else:
            if filename not in self.not_solved:
                self.not_solved.append(filename)
    
    def get_summary(self) -> dict:
        """Get experiment summary statistics."""
        percentage = self.correct_count / self.total_count if self.total_count > 0 else 0
        return {
            "solved_correctly": self.correct_count,
            "percentage": percentage,
            "total": self.total_count,
            "not_solved": self.not_solved
        }
    
    def format_summary_text(self, method: str, problem: str) -> str:
        """Format summary as text for file output."""
        summary = self.get_summary()
        lines = [
            method,
            problem,
            f"Solved correctly: {summary['solved_correctly']}",
            f"Percentage solved correctly (%): {summary['percentage']}",
            f"Total: {summary['total']}",
            ""
        ]
        
        if summary['not_solved']:
            lines.append("Not solved:")
            lines.extend(summary['not_solved'])
        
        return "\n".join(lines)