import json
import tempfile
import unittest
from pathlib import Path
import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from app import run_specialist, Settings, DiscoveryResult, RunStore


class ProviderProtocolTests(unittest.TestCase):
    def test_openai_tool_calls_and_structured_output_round_trip(self):
        requests = []
        @tool
        def search_web(query: str) -> dict:
            """Return a scripted web result for protocol verification."""
            return {'sources': [{'id': 'source-one', 'text': 'Atlas offers cloud tools.'}]}
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                name, args = 'search_web', {'query': 'Target cloud competitors'}
            else:
                name, args = 'DiscoveryResult', {'scope': 'Cloud tools', 'rationale': 'Source describes cloud tools.',
                    'competitors': [{'name': 'Atlas', 'rationale': 'Comparable cloud tools.', 'source_ids': ['source-one']}], 'gaps': ['Only one competitor in fixture.']}
            return httpx.Response(200, json={'id': 'test-response', 'object': 'chat.completion', 'created': 0,
                'model': 'gpt-4.1', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                    'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-' + str(len(requests)),
                        'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}]})
        with tempfile.TemporaryDirectory() as folder, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model='gpt-4.1', api_key='test', base_url='https://api.openai.com/v1',
                use_responses_api=False, http_client=client)
            store = RunStore(Path(folder)/'ledger.sqlite', Settings(company='Target'))
            result = run_specialist('discovery', model, [search_web], DiscoveryResult, {'company': 'Target'}, store)
            self.assertEqual(result.competitors[0].name, 'Atlas')
            self.assertEqual(len(requests), 2)
            self.assertTrue({'search_web', 'DiscoveryResult'} <= {t['function']['name'] for t in requests[0]['tools']})
            self.assertTrue(any(m['role'] == 'tool' for m in requests[1]['messages']))

