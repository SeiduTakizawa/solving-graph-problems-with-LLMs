"""
Main experiment runner with clean, modular architecture.
"""
import os
from typing import List, Union, Dict, Any
from tqdm import tqdm

from .experiment_config import ExperimentConfig, PathManager
from .prompt_builder import PromptBuilder
from .result_processor import ResultProcessor, ExperimentResultCollector
from .file_io import FileManager, GraphDataLoader
from .utils import get_openai_response, get_graph_reasoning


class SingleExperimentRunner:
    """Runs a single experiment on one graph file."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.path_manager = PathManager(config)
        self.prompt_builder = PromptBuilder(config)
        self.result_processor = ResultProcessor(config.problem)
    
    def run_single_file(self, filename: str, graph_loader: GraphDataLoader) -> Dict[str, Any]:
        """Run experiment on a single graph file."""
        # Load data
        is_directed = (self.config.type == "dag")
        graph, questions = graph_loader.load_experiment_data(filename, self.config.problem, is_directed)
        
        if questions is None:
            return None  # Skip if no questions for this problem
        
        # Build prompts
        formatted_questions = self.prompt_builder.build_prompt(questions)
        
        # Get edgelist representation
        edgelist = graph_loader.get_edgelist(graph)
        
        # Query the model
        model_response = self._query_model(formatted_questions)
        
        # Handle structured vs regular responses
        if self.config.structured_output and isinstance(model_response, tuple):
            answers, raw_json_data = model_response
        else:
            answers = model_response
            raw_json_data = None
        
        # Process results
        return self._process_results(filename, questions, formatted_questions, answers, edgelist, raw_json_data)
    
    def _query_model(self, questions: Union[str, List[str]]) -> Union[str, List[str], tuple]:
        """Query the language model with questions."""
        if isinstance(questions, list):
            answers = []
            raw_json_answers = []
            for question in questions:
                if self.config.structured_output:
                    try:
                        answer = get_graph_reasoning(question, self.config)
                        if answer and answer.steps:
                            # Store raw JSON
                            raw_json_answers.append(answer.model_dump())
                            # Convert structured response to string for compatibility
                            answer_str = f"Steps:\n"
                            for i, step in enumerate(answer.steps, 1):
                                answer_str += f"{i}. {step.explanation}\nOutput: {step.output}\n\n"
                            answer_str += f"Final Answer: {answer.final_answer}"
                            answers.append(answer_str)
                        else:
                            # Fallback to regular response if structured parsing fails
                            raw_json_answers.append(None)
                            answer = get_openai_response(question, self.config)
                            answers.append(answer)
                    except Exception as e:
                        print(f"Structured output failed, falling back to regular response: {e}")
                        raw_json_answers.append(None)
                        answer = get_openai_response(question, self.config)
                        answers.append(answer)
                else:
                    raw_json_answers.append(None)
                    answer = get_openai_response(question, self.config)
                    answers.append(answer)
            if self.config.structured_output:
                return answers, raw_json_answers
            return answers
        else:
            if self.config.structured_output:
                try:
                    answer = get_graph_reasoning(questions, self.config)
                    if answer and answer.steps:
                        # Store raw JSON
                        raw_json = answer.model_dump()
                        # Convert structured response to string for compatibility
                        answer_str = f"Steps:\n"
                        for i, step in enumerate(answer.steps, 1):
                            answer_str += f"{i}. {step.explanation}\nOutput: {step.output}\n\n"
                        answer_str += f"Final Answer: {answer.final_answer}"
                        return answer_str, raw_json
                    else:
                        # Fallback to regular response if structured parsing fails
                        return get_openai_response(questions, self.config), None
                except Exception as e:
                    print(f"Structured output failed, falling back to regular response: {e}")
                    return get_openai_response(questions, self.config), None
            else:
                return get_openai_response(questions, self.config)
    
    def _process_results(self, filename: str, original_questions: Union[str, List[str]], 
                        formatted_questions: Union[str, List[str]], 
                        answers: Union[str, List[str]], edgelist, raw_json_data=None) -> Dict[str, Any]:
        """Process and validate results."""
        # Compute ground truth
        ground_truth = self.result_processor.compute_ground_truth(
            edgelist, original_questions if isinstance(original_questions, list) else None
        )
        
        # Parse model answers
        parsed_answers = self.result_processor.parse_answer(answers, edgelist)
        
        # Handle special cases for MST and topological sorting
        if self.config.problem in ["mst", "topological_sorting"]:
            if self.config.problem == "mst":
                from .utils import mst_get_answer
                parsed_answers, ground_truth = mst_get_answer(answers, edgelist)
            elif self.config.problem == "topological_sorting":
                from .utils import topological_get_answer
                parsed_answers, ground_truth = topological_get_answer(answers, edgelist)
        
        # Handle connected_nodes special case
        if self.config.problem == "connected_nodes" and isinstance(original_questions, list):
            parsed_answers = []
            ground_truths = []
            for i, question in enumerate(original_questions):
                node = self.result_processor._extract_node_from_question(question)
                from .utils import connected_nodes_get_answer
                parsed_ans, gt = connected_nodes_get_answer(answers[i], edgelist, node)
                parsed_answers.append(parsed_ans)
                ground_truths.append(gt)
            ground_truth = ground_truths
        
        # Validate results
        result_info = self._validate_and_format_results(
            filename, original_questions, formatted_questions, answers, 
            parsed_answers, ground_truth, raw_json_data
        )
        
        return result_info
    
    def _validate_and_format_results(self, filename: str, original_questions, 
                                   formatted_questions, answers, parsed_answers, 
                                   ground_truth, raw_json_data=None) -> Dict[str, Any]:
        """Validate results and format for saving."""
        if isinstance(original_questions, list):
            # Multi-question case
            results = []
            all_correct = True
            
            for i in range(len(original_questions)):
                is_correct = (parsed_answers[i] == ground_truth[i])
                if not is_correct:
                    all_correct = False
                
                results.append({
                    'correct': "Correct" if is_correct else "False",
                    'ground_truth': ground_truth[i],
                    'found_answer': parsed_answers[i],
                    'question': formatted_questions[i],
                    'answer': answers[i]
                })
            
            return {
                'filename': filename,
                'is_correct': all_correct,
                'results': results,
                'multi_question': True,
                'raw_json_data': raw_json_data
            }
        else:
            # Single question case
            is_correct = (parsed_answers == ground_truth)
            
            return {
                'filename': filename,
                'is_correct': is_correct,
                'correct': "Correct" if is_correct else "False",
                'ground_truth': ground_truth,
                'found_answer': parsed_answers,
                'question': formatted_questions,
                'answer': answers,
                'multi_question': False,
                'raw_json_data': raw_json_data
            }


class ExperimentRunner:
    """Main experiment runner that coordinates all components."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.path_manager = PathManager(config)
        self.single_runner = SingleExperimentRunner(config)
        self.result_collector = ExperimentResultCollector()
    
    def run_experiment(self) -> float:
        """Run the complete experiment."""
        print(self.config)
        
        if not os.path.isdir(self.path_manager.graphs_dir):
            print(f"Skipping: no graph data at {self.path_manager.graphs_dir}")
            return 0
        
        # Setup paths and directories
        self.path_manager.ensure_directories_exist()
        
        # Initialize data loader
        graph_loader = GraphDataLoader(
            self.path_manager.graphs_dir,
            self.path_manager.questions_dir
        )
        
        # Get list of files to process
        files = graph_loader.get_available_files()
        numbered_files = [f"{i}.txt" for i in range(len(files))]
        
        # Process each file
        for i in tqdm(range(len(numbered_files))):
            filename = numbered_files[i]
            
            result_info = self.single_runner.run_single_file(filename, graph_loader)
            
            if result_info is None:
                continue  # Skip files without questions for this problem
            
            # Save detailed results
            self._save_result_file(result_info)
            
            # Update statistics
            self.result_collector.add_result(filename, result_info['is_correct'])
            
            if result_info['is_correct']:
                print("Correct")
        
        if self.result_collector.total_count == 0:
            # Don't write a summary, or --all would treat this run as done.
            print(f"Skipping: no '{self.config.problem}' questions in {self.path_manager.questions_dir}")
            return 0
        
        # Save experiment summary
        self._save_experiment_summary()
        
        # Print and return results
        summary = self.result_collector.get_summary()
        print(self.config)
        print(f"Solved correctly: {summary['solved_correctly']}")
        print(f"Percentage solved correctly (%): {summary['percentage']}")
        
        return summary['percentage']
    
    def _save_result_file(self, result_info: Dict[str, Any]):
        """Save detailed result file."""
        result_path = self.path_manager.get_result_file(result_info['filename'])
        
        # Save regular text results
        if result_info['multi_question']:
            FileManager.save_detailed_result(result_path, result_info['results'])
        else:
            FileManager.save_single_result(
                result_path,
                result_info['correct'],
                result_info['ground_truth'],
                result_info['found_answer'],
                result_info['question'],
                result_info['answer']
            )
        
        # Save JSON results if structured output was used
        if result_info.get('raw_json_data') is not None:
            FileManager.save_json_result(result_path, result_info['raw_json_data'])
    
    def _save_experiment_summary(self):
        """Save experiment summary."""
        summary_text = self.result_collector.format_summary_text(
            self.config.method, self.config.problem
        )
        
        summary_path = self.path_manager.get_experiment_summary_file()
        FileManager.save_experiment_summary(summary_path, summary_text)


def run_single_experiment(config: ExperimentConfig) -> float:
    """Run a single experiment with the given configuration."""
    runner = ExperimentRunner(config)
    return runner.run_experiment()


def run_all_experiments():
    """Run all experiments in batch mode."""
    from .experiment_config import get_all_experiment_configs
    
    configs = get_all_experiment_configs()
    
    for config in configs:
        path_manager = PathManager(config)
        summary_path = path_manager.get_experiment_summary_file()
        
        # Skip if experiment already completed
        if FileManager.file_exists(summary_path):
            continue
        
        print(f"Running experiment: {config.problem}/{config.method}/{config.size}")
        run_single_experiment(config)