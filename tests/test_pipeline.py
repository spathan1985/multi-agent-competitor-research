import tempfile
import unittest
from pathlib import Path
from langgraph.types import Command
from app import ResearchSession, Settings, RunStore


class PipelineTests(unittest.TestCase):
    def test_three_agents_use_tools_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(company="Northstar", mode="offline")
            with ResearchSession(folder, settings) as session:
                session.run(session.initial_input())
                self.assertIsNotNone(session.pending())
                data = session.store.snapshot()
                self.assertEqual(len(data["candidates"]), 3)
                self.assertFalse(data["research"])
                self.assertFalse(data["approved"])
            with ResearchSession(folder, settings) as session:
                session.run(Command(resume={"competitor_ids": ["competitor-1", "competitor-2"]}))
                data = session.store.snapshot()
                self.assertEqual(set(data["analysis"]), {"competitor-1", "competitor-2"})
                self.assertEqual(set(data["research"]), {"competitor-1", "competitor-2"})
                self.assertIsNotNone(data["report"])
                roles = {e["agent"] for e in data["events"] if e["kind"] == "agent_start"}
                self.assertEqual(roles, {"discovery", "research", "analysis"})
                tools = {e["name"] for e in data["events"] if e["kind"] == "tool_call"}
                self.assertTrue({"inspect_sources", "search_web", "search_news"} <= tools)
                self.assertEqual(data["report"]["mode"], "offline")
                self.assertTrue(Path(folder, "briefing.md").is_file())
                self.assertTrue(all("source_ids" in item for result in data["report"]["results"] for item in result["findings"]))
            with ResearchSession(folder, settings) as session:
                self.assertIsNotNone(session.store.snapshot()["report"])
                self.assertIsNone(session.pending())

    def test_reject_shortlist_performs_no_competitor_searches(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company="Northstar", mode="offline")) as session:
            session.run(session.initial_input())
            session.run(Command(resume={"competitor_ids": []}))
            data = session.store.snapshot()
            self.assertFalse(data["research"])
            self.assertEqual(data["report"]["status"], "rejected")
            self.assertEqual(set(data["calls"]), {"discovery"})

    def test_gates_and_budget_persist(self):
        with tempfile.TemporaryDirectory() as folder:
            store = RunStore(Path(folder)/"ledger.sqlite", Settings(company="Target", mode="offline"))
            with self.assertRaises(ValueError): store.reserve("competitor-1", "general")
            for _ in range(4): store.reserve("discovery", "general")
            reopened = RunStore(Path(folder)/"ledger.sqlite", Settings(company="Target", mode="offline"))
            with self.assertRaises(ValueError): reopened.reserve("discovery", "general")
            self.assertEqual(reopened.snapshot()["calls"]["discovery"], 4)

    def test_sources_dates_and_ownership_are_validated(self):
        from app import validate_findings
        from app import AnalysisResult
        sources = [{"id": "s", "url": "https://example.com", "text": "Launch", "published_at": "2026-10-01", "topic": "news"}]
        draft = AnalysisResult(pricing={"text": "Invented", "source_ids": ["unknown"]}, features=[], positioning=None,
            news=[{"text": "Mismatched date", "source_ids": ["s"], "published_at": "2026-10-02"},
                  {"text": "Valid launch", "source_ids": ["s"], "published_at": "2026-10-01"}], gaps=[])
        result = validate_findings(draft, sources, "2026-04-25", "2026-10-25")
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["text"], "Valid launch")
        self.assertTrue(result["gaps"])
