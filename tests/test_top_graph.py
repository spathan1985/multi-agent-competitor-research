import tempfile
import unittest
from pathlib import Path
from langgraph.types import Command
from app import ResearchSession, Settings, RunStore


class TopGraphTests(unittest.TestCase):
    def test_explicit_nodes_execute_and_share_state(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            required = {'discovery', 'approval', 'research', 'analysis', 'compile'}
            self.assertTrue(required <= set(session.graph.get_graph().nodes))
            events = list(session.stream(session.initial_input()))
            self.assertTrue(session.pending())
            events += list(session.stream(Command(resume={'competitor_ids': ['competitor-1', 'competitor-2']})))
            executed = {node for mode, update in events if mode == 'updates' for node in update}
            self.assertTrue(required <= executed)
            state = session.graph.get_state(session.config).values
            self.assertEqual(state['approved'], ['competitor-1', 'competitor-2'])
            self.assertEqual(set(state['research_results']), {'competitor-1', 'competitor-2'})
            self.assertEqual(set(state['analysis_results']), {'competitor-1', 'competitor-2'})
            self.assertIsNotNone(state['report'])

    def test_legacy_shortlist_reopens_at_approval_without_model_call(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(company='Target')
            store = RunStore(Path(folder)/'ledger.sqlite', settings)
            source = store.add_source('discovery', {'url': 'https://example.com', 'content': 'Atlas competitor'}, 'general')
            store.update(lambda d: d.update(discovery_done=True, scope='Cloud', candidates=[{
                'id': 'competitor-1', 'name': 'Atlas', 'rationale': 'Comparable offering', 'source_ids': [source['id']]}]))
            model = Mock()
            with ResearchSession(folder, settings, model=model) as session:
                self.assertIsNotNone(session.pending())
                self.assertEqual(session.pending()['competitors'][0]['name'], 'Atlas')
                model.bind_tools.assert_not_called()
                self.assertFalse(store.snapshot()['research'])
            with ResearchSession(folder, settings) as session:
                session.run(Command(resume={'competitor_ids': []}))
                self.assertEqual(store.snapshot()['report']['status'], 'rejected')


    def test_no_model_orchestrator_and_state_drives_routing(self):
        from app import ScriptedChatModel
        class SpecialistsOnly(ScriptedChatModel):
            def _generate(self, messages, **kwargs):
                assert str(messages[0].content).splitlines()[0] in ('ROLE=discovery', 'ROLE=research', 'ROLE=analysis')
                return super()._generate(messages, **kwargs)
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target'), model=SpecialistsOnly()) as session:
            self.assertNotIn('orchestrator', session.graph.get_graph().nodes)
            session.run(session.initial_input())
            session.run(Command(resume={'competitor_ids': ['competitor-1', 'competitor-2']}))
            state = session.graph.get_state(session.config).values
            self.assertEqual(state['report']['status'], 'complete')
            self.assertEqual(set(state['analysis_results']), set(state['approved']))
            self.assertNotIn('orchestrator_calls', session.store.snapshot())

    def test_saved_research_skips_discovery_and_search_on_resume(self):
        from app import ScriptedChatModel
        class AnalysisOnly(ScriptedChatModel):
            def _generate(self, messages, **kwargs):
                assert str(messages[0].content).startswith('ROLE=analysis')
                return super()._generate(messages, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(company='Target')
            store = RunStore(Path(folder)/'ledger.sqlite', settings)
            source = store.add_source('discovery', {'url': 'https://example.com', 'content': 'Atlas competitor'}, 'general')
            store.update(lambda d: d.update(discovery_done=True, candidates=[{
                'id': 'competitor-1', 'name': 'Atlas', 'rationale': 'Comparable', 'source_ids': [source['id']]}]))
            store.approve(['competitor-1'])
            store.add_source('competitor-1', {'url': 'https://example.com/product', 'content': 'Shared documents for teams'}, 'general')
            store.update(lambda d: d['research'].update({'competitor-1': {'summary': 'Saved collection', 'gaps': []}}))
            before = store.snapshot()['calls']
            with ResearchSession(folder, settings, model=AnalysisOnly()) as session:
                session.run(None)
                self.assertTrue(session.store.snapshot()['analysis']['competitor-1']['findings'])
                self.assertEqual(before, session.store.snapshot()['calls'])
                self.assertIsNotNone(session.graph.get_state(session.config).values['report'])
