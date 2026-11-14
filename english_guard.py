"""Reject untranslated engineering prose in tracked source and documentation.

Only reviewed, exact source-language data literals can be allowed by path and
SHA-256. Comments, docstrings and file names never inherit those exceptions.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize

CYRILLIC = re.compile(r'[\u0400-\u04ff]')

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def inspect_text(path, text, policy):
    allowed = {entry['sha256'] for entry in policy.get('data_literals', []) if entry['path'] == path}
    findings = []
    if CYRILLIC.search(path):
        findings.append({'path': path, 'line': 1, 'kind': 'file-name'})
    if path.endswith('.py'):
        try:
            tree = ast.parse(text)
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        docs.add((first.value.lineno, first.value.col_offset))
            for item in tokenize.generate_tokens(io.StringIO(text).readline):
                if item.type == tokenize.COMMENT and CYRILLIC.search(item.string):
                    findings.append({'path': path, 'line': item.start[0], 'kind': 'comment'})
                elif item.type == tokenize.STRING:
                    try:
                        value = ast.literal_eval(item.string)
                    except (ValueError, SyntaxError):
                        value = item.string
                    if isinstance(value, str) and CYRILLIC.search(value):
                        is_doc = (item.start[0], item.start[1]) in docs
                        if is_doc or digest(value) not in allowed:
                            findings.append({'path': path, 'line': item.start[0], 'kind': 'docstring' if is_doc else 'string'})
                # Python 3.12+ exposes f-string literal text as distinct tokens.
                elif tokenize.tok_name.get(item.type) == 'FSTRING_MIDDLE' and CYRILLIC.search(item.string):
                    findings.append({'path': path, 'line': item.start[0], 'kind': 'fstring'})
        except (SyntaxError, tokenize.TokenError) as error:
            findings.append({'path': path, 'line': getattr(error, 'lineno', 1) or 1, 'kind': 'syntax'})
    elif path.endswith('.json'):
        try:
            def walk(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if CYRILLIC.search(key) and digest(key) not in allowed:
                            findings.append({'path': path, 'line': 1, 'kind': 'json-key'})
                        walk(child)
                elif isinstance(value, list):
                    for child in value:walk(child)
                elif isinstance(value, str) and CYRILLIC.search(value) and digest(value) not in allowed:
                    findings.append({'path': path, 'line': 1, 'kind': 'json-value'})
            walk(json.loads(text))
        except ValueError:
            findings.append({'path': path, 'line': 1, 'kind': 'json-syntax'})
    else:
        for number, line in enumerate(text.splitlines(), 1):
            if CYRILLIC.search(line):findings.append({'path': path, 'line': number, 'kind': 'prose'})
    return findings

def main():
    root = Path(__file__).resolve().parent
    policy = json.loads((root / 'english_policy.json').read_text(encoding='utf-8'))
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    findings = []; checked = 0
    for path in filter(None, paths):
        try:text = (root / path).read_text(encoding='utf-8-sig')
        except (OSError, UnicodeError):continue
        if '\0' in text:continue
        checked += 1;findings.extend(inspect_text(path, text, policy))
    print(json.dumps({'checked_files': checked, 'findings': findings}))
    return bool(findings)

if __name__ == '__main__':
    raise SystemExit(main())
