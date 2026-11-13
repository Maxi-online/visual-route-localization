import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('publication_guard',ROOT/'publication_guard.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

class PublicationPrivacyTests(unittest.TestCase):
    def test_literal_fallback_in_boolean_expression_is_detected(self):
        found=guard.python_findings("api_key = options.key or 'opaque-credential-value'")
        self.assertEqual(len(found),1)

    def test_conditional_and_annotated_defaults_are_detected(self):
        for code in ["api_key: str = 'opaque-credential-value'", "api_key = selected if selected else 'opaque-credential-value'"]:
            self.assertEqual(len(guard.python_findings(code)),1)

    def test_dictionary_and_function_default_are_detected(self):
        for code in ["headers = {'pass': 'opaque-credential-value'}", "def connect(api_key='opaque-credential-value'): pass"]:
            self.assertEqual(len(guard.python_findings(code)),1)

    def test_configuration_file_secret_is_detected(self):
        found=guard.inspect_text('config/settings.json','{"provider":{"api_key":"opaque-credential-value"}}',{})
        self.assertEqual(found[0]['kind'],'configuration-credential')
        self.assertNotIn('value',found[0])

    def test_environment_lookup_and_empty_value_are_safe(self):
        self.assertEqual(guard.python_findings("api_key = os.environ.get('OPENAI_API_KEY', '')"),[])
        self.assertEqual(guard.structured_findings({'api_key':'${OPENAI_API_KEY}'},''),[])

    def test_enum_label_is_not_a_password(self):
        self.assertEqual(guard.python_findings("class Outcome(str, Enum):\n    PASS = 'pass'"),[])

    def test_private_identifier_is_detected_without_echoing_it(self):
        policy={'blocked_identifier_sha256':[guard.digest('tenant.example.invalid')]}
        found=guard.inspect_text('config/settings.yaml','host: tenant.example.invalid',policy)
        self.assertTrue(any(item['kind']=='private-identifier' for item in found))
        self.assertNotIn('tenant.example.invalid',str(found))


    def test_non_env_configuration_formats_are_checked(self):
        for name,text in [
            ('config/secrets.ini','[provider]\napi_key=opaque-credential-value'),
            ('config/secrets.toml','[provider]\napi_key="opaque-credential-value"'),
            ('config/secrets.xml','<provider><api_key>opaque-credential-value</api_key></provider>'),
            ('settings.json.example','{"api_key":"opaque-credential-value"}'),
            ('config/settings.yaml','provider:\n  api_key: opaque-credential-value'),
        ]:
            found=guard.inspect_text(name,text,{})
            self.assertTrue(any(item['kind']=='configuration-credential' for item in found),name)
