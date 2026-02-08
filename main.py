"""Entry point for the multi-agent research system.

Usage:
    python main.py "your research question"
    python main.py          # interactive mode
"""

import sys

from graph.state import create_initial_state
from graph.workflow import compile_workflow
from config import settings


def run_research(query: str, verbose: bool = True) -> str:
    """Run the full research workflow and return the final report."""
    if verbose:
        print(f"Query: {query}")

    initial_state = create_initial_state(query)
    workflow = compile_workflow(checkpointing=True)
    config = {"configurable": {"thread_id": "research-1"}}

    if verbose:
        print("Starting research workflow\n")

    final_state = None
    for event in workflow.stream(initial_state, config):
        if verbose:
            for node_name in event:
                print(f"  completed: {node_name}")
        final_state = event

    for node_state in final_state.values():
        if isinstance(node_state, dict) and "final_report" in node_state:
            return node_state["final_report"]

    return "Research completed but no report was generated."


def interactive_mode():
    print("Interactive research mode (type 'quit' to exit)\n")

    while True:
        try:
            query = input("Research query: ").strip()

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not query:
                print("Please enter a query.\n")
                continue

            report = run_research(query)
            print("\n" + "-" * 60)
            print("RESEARCH REPORT")
            print("-" * 60)
            print(report)
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    if not settings.google_api_key:
        print("Error: GOOGLE_API_KEY not set.")
        print("Create a .env file with your key (see .env.example).")
        print("Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        report = run_research(query)
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60)
        print(report)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
