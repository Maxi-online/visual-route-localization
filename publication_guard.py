"""Review tracked configuration and Python credential literals without echoing values.

This complements a native secret scanner. It is a regression check, not proof
that arbitrary source code or an unknown credential format cannot leak data.
"""
import ast
import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
import yaml

SECRET_NAME = re.compile(r'api.?key|access.?key|client.?secret|private.?key|secret.?key|(^|[_.-])secret($|[_.-])|password|passwd|pwd$|(?:^|[_.-])pass$|(^|[_.-])(?:bot.?|auth.?|access.?|refresh.?|service.?)?token($|[_.-])|authorization|database_url|redis_url|proxy_url', re.I)
SKIP_NAME = re.compile(r'token_(?:type|url|count|name|prefix|expires|valid|set|data|re|pattern|expiry)|tokens?_(?:limit|budget)|max_.*tokens?|password_hasher|(?:key|token|secret)_(?:path|file)|ssl_private_key_path|_TOKEN_RE|\.(?:json|ya?ml)$',re.I)
URL = re.compile(r'\b(?:https?|wss?|postgres(?:ql)?|redis|mongodb(?:\+srv)?|amqp)://[^\s<>\x22\x27`\]\[)]+',re.I)
ATOM = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{2,}')

def digest(value):
    return hashlib.sha256(value.casefold().encode('utf-8')).hexdigest()

def safe_literal(value):
    if not value.strip():return True
    if '${' in value or '{{' in value:return True
    if re.fullmatch(r'\$[A-Z_][A-Z0-9_]*',value):return True
    if value.lower() in {'none','null','true','false','str','string','bearer','basic','password','secret','token','api_key'}:return True
    if re.search(r'(?i)your[_ -]|replace[_ -]|placeholder|change.?me|dummy[_ -]|demo[_ -]|mock[_ -]|offline-test-|unit-test-|<[^>]+>',value):return True
    if re.search(r'\{[A-Za-z_][^}]*\}',value):return True
    try:
        parsed=urlsplit(value)
        if parsed.scheme and parsed.hostname and parsed.password is None:return True
    except ValueError:pass
    return False

def literal_values(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):yield node
    elif isinstance(node,ast.BoolOp):
        for value in node.values:yield from literal_values(value)
    elif isinstance(node,ast.IfExp):
        yield from literal_values(node.body);yield from literal_values(node.orelse)
    elif isinstance(node,ast.Call):
        name=ast.unparse(node.func)
        if name.endswith(('.get','.getenv')) or name in {'getenv','get'}:
            if len(node.args)>1:yield from literal_values(node.args[1])
        elif name in {'Field','str','SecretStr'} and node.args:yield from literal_values(node.args[0])
        for kw in node.keywords:
            if kw.arg in {'default','default_value'}:yield from literal_values(kw.value)

def python_findings(text):
    tree=ast.parse(text);found=[];seen=set();parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    def add(name,node):
        if not SECRET_NAME.search(name) or SKIP_NAME.search(name):return
        owner=node
        while owner in parents:
            owner=parents[owner]
            if isinstance(owner,ast.ClassDef) and any(ast.unparse(base).endswith('Enum') for base in owner.bases):return
        for leaf in literal_values(node):
            if len(leaf.value)<3 or safe_literal(leaf.value):continue
            key=(leaf.lineno,leaf.value)
            if key in seen:continue
            seen.add(key);found.append({'line':leaf.lineno,'kind':'literal-credential','value':leaf.value})
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:add(ast.unparse(target),node.value)
        elif isinstance(node,ast.AnnAssign) and node.value is not None:add(ast.unparse(node.target),node.value)
        elif isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and isinstance(key.value,str):add(key.value,value)
        elif isinstance(node,ast.keyword) and node.arg:add(node.arg,node.value)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            args=[*node.args.posonlyargs,*node.args.args]
            for arg,value in zip(args[-len(node.args.defaults):],node.args.defaults):add(arg.arg,value)
            for arg,value in zip(node.args.kwonlyargs,node.args.kw_defaults):
                if value is not None:add(arg.arg,value)
    return found

def structured_findings(data,text,trail=()):
    found=[]
    if isinstance(data,dict):
        for key,value in data.items():
            path=(*trail,str(key))
            if isinstance(value,str):
                credential=bool(SECRET_NAME.search(str(key)) and not SKIP_NAME.search(str(key)))
                if str(key).lower() in {'key','value'} and any(part.lower() in {'api','credentials','secrets','openai','anthropic'} or SECRET_NAME.search(part) for part in trail):credential=True
                # GitHub permission levels are capabilities, not token values.
                if trail and trail[-1]=='permissions':credential=False
                if credential and len(value)>=3 and not safe_literal(value):
                    offset=text.find(value);found.append({'line':text.count('\n',0,max(0,offset))+1,'kind':'configuration-credential','value':value})
            elif isinstance(value,(dict,list)):found.extend(structured_findings(value,text,path))
    elif isinstance(data,list):
        for index,value in enumerate(data):found.extend(structured_findings(value,text,(*trail,str(index))))
    return found

def text_identifiers(text):
    atoms=list(ATOM.finditer(text))
    for atom in atoms:yield atom.group(),atom.start()
    for first,second in zip(atoms,atoms[1:]):
        if text[first.end():second.start()].strip()=='':yield first.group()+' '+second.group(),first.start()
    for match in URL.finditer(text):
        yield match.group(),match.start()
        try:host=urlsplit(match.group()).hostname
        except ValueError:continue
        if host:yield host,match.start()

def inspect_text(path,text,policy):
    findings=[]
    if path.endswith('.py'):
        try:findings.extend(python_findings(text))
        except SyntaxError:findings.append({'line':1,'kind':'invalid-python','value':''})
    if path.endswith(('.json','.yaml','.yml','.json.example','.yaml.example','.yml.example','.toml','.ini','.cfg','.xml')):
        try:
            if path.endswith(('.json','.json.example')):data=json.loads(text)
            elif path.endswith('.toml'):data=tomllib.loads(text)
            elif path.endswith(('.ini','.cfg')):
                parser=configparser.ConfigParser(interpolation=None)
                parser.read_string(text)
                data={section:dict(parser.items(section)) for section in parser.sections()}
            elif path.endswith('.xml'):
                data=None
                for element in ET.fromstring(text).iter():
                    if element.text and element.text.strip():findings.extend(structured_findings({element.tag:element.text.strip()},text))
            else:data=yaml.safe_load(text)
            if data is not None:findings.extend(structured_findings(data,text))
        except (ValueError,yaml.YAMLError,configparser.Error,ET.ParseError):findings.append({'line':1,'kind':'invalid-configuration','value':''})
    blocked=set(policy.get('blocked_identifier_sha256',[]))
    for value,offset in text_identifiers(text):
        if digest(value) in blocked:findings.append({'line':text.count('\n',0,offset)+1,'kind':'private-identifier','value':value})
    approved={(item['path'],item['sha256']) for item in policy.get('reviewed_non_credentials',[])}
    return [{'path':path,'line':item['line'],'kind':item['kind']} for item in findings
            if (path,digest(item['value'])) not in approved]

def main():
    root=Path(__file__).resolve().parent
    policy=json.loads((root/'publication_policy.json').read_text(encoding='utf-8'))
    names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    findings=[];count=0
    for name in filter(None,names):
        raw=(root/name).read_bytes()
        try:text=raw.decode('utf-16') if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-8-sig')
        except UnicodeError:continue
        if '\0' in text:continue
        count+=1;findings.extend(inspect_text(name,text,policy))
    print(json.dumps({'tracked_text_files_checked':count,'findings':findings},indent=2))
    return bool(findings)

if __name__=='__main__':sys.exit(main())
