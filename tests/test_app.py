import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_discover_approve_and_render_all_sections(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'COMPETITOR_RUN_ROOT': folder}):
            app = AppTest.from_file(str(Path.cwd() / 'app.py'), default_timeout=20).run()
            self.assertFalse(app.exception)
            app.text_input(key='company').set_value('Northstar')
            app.button(key='start').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.multiselect(key='approved').options), 3)
            app.button(key='approve').click().run()
            self.assertFalse(app.exception)
            self.assertEqual([s.value for s in app.subheader].count('Pricing'), 3)
            self.assertTrue(any('**Business plan:**' in m.value and '\n\n[Source 1' in m.value for m in app.markdown))
            self.assertTrue(any('SYNTHETIC' in m.value for m in app.info))
            self.assertEqual(len(list(Path(folder).glob('*/briefing.json'))), 1)

    def test_markdown_preserves_formatting_and_resolves_citations(self):
        from app import finding_markdown
        text = finding_markdown({'text': '**Plan** costs $20 【s】', 'source_ids': ['s']},
                                {'s': {'url': 'https://example.com/pricing'}})
        self.assertIn('**Plan**', text)
        self.assertIn(r'\$20', text)
        self.assertNotIn('【', text)
        self.assertIn('\n\n[Source 1', text)
