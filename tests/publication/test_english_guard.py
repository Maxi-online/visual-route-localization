"""Exercise the language boundary without banning legitimate corpus data."""
import unittest
from english_guard import digest, inspect_text

class EnglishBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.word = ''.join(chr(code) for code in (0x442, 0x435, 0x441, 0x442))
        self.policy = {'data_literals': [{'path': 'parser.py', 'sha256': digest(self.word), 'reason': 'Input vocabulary'}]}

    def test_untranslated_comments_and_docstrings_fail(self):
        self.assertTrue(inspect_text('parser.py', '# ' + self.word, self.policy))
        self.assertTrue(inspect_text('parser.py', '"""' + self.word + '"""', self.policy))

    def test_exact_linguistic_literal_is_allowed_only_in_reviewed_path(self):
        source = 'keyword = ' + repr(self.word)
        self.assertFalse(inspect_text('parser.py', source, self.policy))
        self.assertTrue(inspect_text('other.py', source, self.policy))

    def test_new_messages_and_escaped_prose_fail(self):
        self.assertTrue(inspect_text('parser.py', 'message = ' + repr(self.word + self.word), self.policy))
        escaped = (self.word + self.word).encode('unicode_escape').decode()
        self.assertTrue(inspect_text('parser.py', 'message = "' + escaped + '"', self.policy))

    def test_documentation_is_checked_without_folder_exclusions(self):
        self.assertTrue(inspect_text('docs/setup.md', self.word, self.policy))
        self.assertFalse(inspect_text('docs/setup.md', 'Run the offline checks.', self.policy))

if __name__ == '__main__':
    unittest.main()
