
"""
Refactored main entry point for graph reasoning experiments.
Clean, modular architecture replacing the original spaghetti code.
"""
import argparse
from graph_reasoning.experiment_config import ExperimentConfig
from graph_reasoning.experiment_runner import run_single_experiment, run_all_experiments


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Graph reasoning experiments")
    parser.add_argument('--adj', type=str, default="edgelist", 
                       help='Adjacency matrix in matrix or list (default: edgelist)')
    parser.add_argument('--problem', type=str, default="node_count", 
                       help='Problem to solve (default: count nodes)')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", 
                       help='Name of LM (default: gpt-3.5-turbo)')
    parser.add_argument('--type', type=str, default="er", 
                       help='Graph type (default: er)')
    parser.add_argument('--size', type=str, default="small", 
                       help='Graph size (default: small)')
    parser.add_argument('--temperature', type=int, default=0, 
                       help='Temperature (default: 0)')
    parser.add_argument('--token', type=int, default=4000, 
                       help='Max token (default: 4000)')
    parser.add_argument('--method', type=str, default="none", 
                       help='Method to experiment (default: none)')
    parser.add_argument('--structured', action='store_true', 
                       help='Use structured JSON output (default: False)')
    parser.add_argument('-a', '--all', action='store_true', 
                       help='Run all experiments (default: run single experiment)')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    else:
        # Create configuration from arguments
        config = ExperimentConfig(
            problem=args.problem,
            adj=args.adj,
            model=args.model,
            type=args.type,
            size=args.size,
            temperature=args.temperature,
            token=args.token,
            method=args.method,
            structured_output=args.structured
        )
        
        run_single_experiment(config)


if __name__ == "__main__":
    main()