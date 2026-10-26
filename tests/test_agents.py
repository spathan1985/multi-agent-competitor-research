import tempfile
import unittest
from uuid import uuid4
from langgraph.types import Command
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app import ScriptedChatModel, Settings, ResearchSession


class RefiningModel(ScriptedChatModel):
    def _generate(self, messages, **kwargs):
        if str(messages[0].content).startswith('ROLE=research'):
            web_calls = [m for m in messages if isinstance(m, ToolMessage) and m.name == 'search_web']
            if len(web_calls) == 1:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=[{
                    'name': 'search_web', 'args': {'query': 'official pricing billing terms clarification'},
                    'id': str(uuid4()), 'type': 'tool_call'}]))])
        return super()._generate(messages, **kwargs)


class AgentTests(unittest.TestCase):
    def test_research_agent_can_choose_an_additional_search(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Northstar'), model=RefiningModel()) as session:
            session.run(session.initial_input())
            session.run(Command(resume={'competitor_ids': ['competitor-1']}))
            data = session.store.snapshot()
            self.assertEqual(data['calls']['competitor-1'], 3)
            self.assertTrue(any('clarification' in e['detail'] for e in data['events']))
            self.assertEqual(data['report']['status'], 'complete')

    def test_node_operations_cannot_bypass_approval_or_compile_early(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            self.assertIn('error', session.compile_briefing())
            self.assertIn('error', session.run_research(competitor_id='competitor-1', instruction='Research now'))
            session.run(session.initial_input())
            with self.assertRaises(ValueError): session.store.approve(['competitor-1', 'competitor-1'])
            session.store.approve(['competitor-1'])
            self.assertIn('error', session.run_analysis(competitor_id='competitor-1', instruction='Analyze now'))
            self.assertIn('error', session.compile_briefing())

    def test_completed_research_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            session.run(session.initial_input())
            session.run(Command(resume={'competitor_ids': ['competitor-1']}))
            before = session.store.snapshot()['calls']
            operation = session.run_research
            operation('competitor-1', 'Again')
            self.assertEqual(before, session.store.snapshot()['calls'])

    def test_tavily_receives_saved_news_window(self):
        from unittest.mock import patch
        from tavily_tools import search_tools
        async def response(**kwargs): return {'results': []}
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            session.run(session.initial_input())
            session.store.approve(['competitor-1'])
            tools = {t.name: t for t in search_tools(session.store, 'competitor-1', 'live')}
            with patch.dict('os.environ', {'TAVILY_API_KEY': 'test'}), patch('langchain_tavily._utilities.TavilySearchAPIWrapper.raw_results_async', side_effect=response) as transport:
                result = tools['search_news'].invoke({'query': 'Atlas product launches'})
                self.assertEqual(result['sources'], [])
                self.assertEqual(transport.call_args.kwargs['start_date'], session.store.snapshot()['news_start'])
                self.assertIn('end_date', transport.call_args.kwargs)

    def test_live_model_configuration_is_openai_and_keys_not_persisted(self):
        from unittest.mock import patch
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'OPENAI_API_KEY': 'secret-test', 'TAVILY_API_KEY': 'tavily-test'}), patch('app.ChatOpenAI', return_value=ScriptedChatModel()) as client:
            with ResearchSession(folder, Settings(company='Target', mode='live')):
                self.assertEqual(client.call_args.kwargs['base_url'], 'https://api.openai.com/v1')
                self.assertFalse(client.call_args.kwargs['use_responses_api'])
                self.assertEqual(client.call_args.kwargs['model'], 'gpt-4.1')
                self.assertEqual(client.call_args.kwargs['api_key'], 'secret-test')
                self.assertNotIn('secret-test', Path(folder, 'settings.json').read_text())


    def test_duplicate_research_calls_execute_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        import time
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            session.run(session.initial_input())
            session.store.approve(['competitor-1'])
            operation = session.run_research
            def research_once(model, store, settings, competitor, instruction):
                time.sleep(0.03)
                store.update(lambda d: d['research'].update({competitor['id']: {'summary': 'Good result', 'gaps': []}}))
            with patch('app.research', side_effect=research_once) as work, ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(operation, 'competitor-1', 'Gather sources') for _ in range(2)]
                for job in jobs: job.result()
                self.assertEqual(work.call_count, 1)

    def test_incomplete_shortlist_is_a_partial_report(self):
        from app import build_report
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            session.run(session.initial_input())
            session.run(Command(resume={'competitor_ids': ['competitor-1']}))
            data = session.store.snapshot()
            data['candidates'] = data['candidates'][:1]
            data['discovery_gaps'] = ['Only one competitor could be supported.']
            report = build_report(data)
            self.assertEqual(report['status'], 'partial')
            from app import markdown_report
            self.assertIn('Only one competitor could be supported.', markdown_report(report))


    def test_eight_searches_leave_room_for_final_research_output(self):
        class EightSearches(ScriptedChatModel):
            def _generate(self, messages, **kwargs):
                if str(messages[0].content).startswith('ROLE=research'):
                    calls = [m for m in messages if isinstance(m, ToolMessage)]
                    if 2 <= len(calls) < 8:
                        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=[{
                            'name': 'search_web', 'args': {'query': 'Targeted followup ' + str(len(calls))},
                            'id': str(uuid4()), 'type': 'tool_call'}]))])
                return super()._generate(messages, **kwargs)
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target'), model=EightSearches()) as session:
            session.run(session.initial_input())
            session.run(Command(resume={'competitor_ids': ['competitor-1']}))
            self.assertEqual(session.store.snapshot()['calls']['competitor-1'], 8)
            self.assertEqual(session.store.snapshot()['report']['status'], 'complete')
            with self.assertRaises(ValueError): session.store.reserve('competitor-1', 'general')
