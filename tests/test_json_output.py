import json
import tempfile
import unittest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from app import ScriptedChatModel, Settings, ResearchSession


class PlainJSONModel(ScriptedChatModel):
    def _generate(self, messages, **kwargs):
        result = super()._generate(messages, **kwargs)
        message = result.generations[0].message
        if message.tool_calls and message.tool_calls[0]['name'] in ('DiscoveryResult', 'ResearchResult', 'AnalysisResult'):
            content = json.dumps(message.tool_calls[0]['args'])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='```json\n' + content + '\n```'))])
        return result


class JSONOutputTests(unittest.TestCase):
    def test_all_specialists_accept_schema_valid_json_messages(self):
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target'), model=PlainJSONModel()) as session:
            session.run(session.initial_input())
            self.assertIsNotNone(session.pending())
            self.assertEqual(len(session.pending()['competitors']), 3)
            session.run(Command(resume={'competitor_ids': ['competitor-1']}))
            self.assertEqual(session.store.snapshot()['report']['status'], 'complete')
            self.assertFalse(any(e['kind'] == 'agent_error' for e in session.store.snapshot()['events']))

    def test_failure_progress_is_explicit_and_can_compile_without_human_rejection(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target')) as session:
            with patch('app.discover', side_effect=ValueError('Invalid result')):
                state = session.run_discovery(instruction='Discover competitors')
            self.assertTrue(session.store.snapshot()['discovery_gaps'])
            from app import next_step
            self.assertEqual(next_step(state), 'compile')
            output = session.compile_briefing()
            self.assertEqual(output.get('status'), 'discovery_failed')
            self.assertFalse(session.store.snapshot()['approval_decided'])

    def test_malformed_json_does_not_create_a_shortlist(self):
        class InvalidJSON(PlainJSONModel):
            def _generate(self, messages, **kwargs):
                if str(messages[0].content).startswith('ROLE=discovery'):
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content='{"competitors": "invented"}'))])
                return super()._generate(messages, **kwargs)
        with tempfile.TemporaryDirectory() as folder, ResearchSession(folder, Settings(company='Target'), model=InvalidJSON()) as session:
            state = session.run_discovery(instruction='Discover')
            self.assertTrue(session.store.snapshot()['discovery_gaps'])
            self.assertFalse(session.store.snapshot()['candidates'])

    def test_prose_final_is_corrected_once_without_repeating_tools(self):
        from unittest.mock import Mock, patch
        from app import run_specialist
        from app import ResearchResult
        from app import RunStore
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            store = RunStore(Path(folder)/'ledger.sqlite', Settings(company='Target'))
            model = Mock()
            model.bind_tools.return_value.invoke.return_value = AIMessage(content='', tool_calls=[
                dict(name='ResearchResult', args={'summary': 'Collected sources', 'gaps': []}, id='repair')])
            with patch('app.create_agent') as factory:
                factory.return_value.invoke.return_value = {'messages': [AIMessage(content='Here is my research summary.')], 'structured_response': None}
                result = run_specialist('research', model, [], ResearchResult, {}, store)
                self.assertEqual(result.summary, 'Collected sources')
                factory.return_value.invoke.assert_called_once()
                model.bind_tools.assert_called_once_with([ResearchResult], tool_choice='ResearchResult')
                model.bind_tools.return_value.invoke.assert_called_once()
