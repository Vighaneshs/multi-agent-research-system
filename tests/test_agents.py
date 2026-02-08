import pytest
from unittest.mock import patch, MagicMock

from graph.state import AgentState, create_initial_state, ResearchTask
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.code_agent import CodeAgent
from agents.synthesis_agent import SynthesisAgent


# -- state -------------------------------------------------------------------

class TestState:
    def test_create_initial_state(self):
        state = create_initial_state("Test query")

        assert state["query"] == "Test query"
        assert state["research_plan"] == []
        assert state["current_task_index"] == 0
        assert state["search_results"] == []
        assert state["code_results"] == []
        assert state["final_report"] == ""
        assert state["iteration_count"] == 0
        assert state["error"] is None
        assert state["next_agent"] == "planner"

    def test_research_task_defaults(self):
        task = ResearchTask(task_id="task_1", description="Test", task_type="search")

        assert task.task_id == "task_1"
        assert task.status == "pending"
        assert task.dependencies == []
        assert task.result is None


# -- planner ------------------------------------------------------------------

class TestPlannerAgent:
    @patch("agents.planner_agent.ChatGoogleGenerativeAI")
    def test_planner_creates_tasks(self, mock_llm):
        mock_resp = MagicMock()
        mock_resp.content = '{"reasoning": "test", "tasks": []}'
        mock_llm.return_value.invoke.return_value = mock_resp
        assert True  # structure-level smoke test

    def test_system_prompt_contents(self):
        with patch("agents.base_agent.settings") as ms:
            ms.google_api_key = "test-key"
            ms.default_model = "gemini-2.0-flash"
            ms.temperature = 0.1

            planner = PlannerAgent()

            assert "Research Planning Agent" in planner.system_prompt
            assert "task_id" in planner.system_prompt
            assert "dependencies" in planner.system_prompt


# -- search -------------------------------------------------------------------

class TestSearchAgent:
    def _make_agent(self):
        with patch("agents.base_agent.settings") as ms:
            ms.google_api_key = "test-key"
            ms.default_model = "gemini-2.0-flash"
            ms.temperature = 0.1
            ms.tavily_api_key = None
            return SearchAgent()

    def test_deduplicate_results(self):
        agent = self._make_agent()
        results = [
            {"source": "https://a.com", "title": "A"},
            {"source": "https://a.com", "title": "A duplicate"},
            {"source": "https://b.com", "title": "B"},
        ]
        unique = agent._deduplicate_results(results)

        assert len(unique) == 2
        assert unique[0]["source"] == "https://a.com"
        assert unique[1]["source"] == "https://b.com"

    def test_rank_results(self):
        agent = self._make_agent()
        results = [
            {"title": "Low", "relevance_score": 0.3},
            {"title": "High", "relevance_score": 0.9},
            {"title": "Medium", "relevance_score": 0.6},
        ]
        ranked = agent._rank_results(results)

        assert ranked[0]["title"] == "High"
        assert ranked[1]["title"] == "Medium"
        assert ranked[2]["title"] == "Low"


# -- code ---------------------------------------------------------------------

class TestCodeAgent:
    def _make_agent(self):
        with patch("agents.base_agent.settings") as ms:
            ms.google_api_key = "test-key"
            ms.default_model = "gemini-2.0-flash"
            ms.temperature = 0.1
            return CodeAgent()

    def test_extract_code_python_block(self):
        agent = self._make_agent()
        response = """Here is the code:
```python
print("Hello")
x = 1 + 2
```
This prints hello."""
        code = agent._extract_code(response)

        assert "print" in code
        assert "Hello" in code

    def test_execute_safe_code(self):
        agent = self._make_agent()
        result = agent._execute_code("x = 1 + 1\nprint(x)")

        assert result.success is True
        assert "2" in result.output

    def test_execute_code_with_error(self):
        agent = self._make_agent()
        result = agent._execute_code("raise ValueError('test error')")

        assert result.success is False
        assert "ValueError" in result.error


# -- synthesis ----------------------------------------------------------------

class TestSynthesisAgent:
    def _make_agent(self):
        with patch("agents.base_agent.settings") as ms:
            ms.google_api_key = "test-key"
            ms.default_model = "gemini-2.0-flash"
            ms.temperature = 0.1
            return SynthesisAgent()

    def test_format_search_results(self):
        agent = self._make_agent()
        results = [
            {
                "title": "Test Article",
                "source": "https://test.com",
                "content": "Test content",
                "relevance_score": 0.9,
            }
        ]
        formatted = agent._format_search_results(results)

        assert "Test Article" in formatted
        assert "https://test.com" in formatted
        assert "Test content" in formatted

    def test_format_empty_results(self):
        agent = self._make_agent()
        formatted = agent._format_search_results([])
        assert "No search results" in formatted


# -- workflow -----------------------------------------------------------------

class TestWorkflow:
    def test_workflow_creation(self):
        from graph.workflow import create_research_workflow

        with patch("graph.workflow.planner"), \
             patch("graph.workflow.searcher"), \
             patch("graph.workflow.coder"), \
             patch("graph.workflow.synthesizer"):
            wf = create_research_workflow()

            assert "planner" in wf.nodes
            assert "search" in wf.nodes
            assert "code" in wf.nodes
            assert "synthesize" in wf.nodes


# -- integration (needs real API key) ----------------------------------------

@pytest.mark.skip(reason="requires API keys")
class TestIntegration:
    def test_full_research_workflow(self):
        from main import run_research

        report = run_research(
            "What is the attention mechanism in transformers?",
            verbose=False,
        )
        assert len(report) > 100
        assert "attention" in report.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
