import re
import os
import sys

# 确保 stdout 用 utf-8
sys.stdout.reconfigure(encoding='utf-8')

base = r'c:\_Notes\AI_Study_Notes\A_基础与范式\01_数学基础'
fp = os.path.join(base, '02_微积分.md')

with open(fp, encoding='utf-8') as f:
    content = f.read()

# 保护 [[...\|...]] wikilink 转义
WIKILINK = re.compile(r'\[\[[^\]]*?\\\|[^\]]*?\]\]')
placeholders = {}

def _sub(m):
    key = f'__WIKILINK_PLACEHOLDER_{len(placeholders)}__'
    placeholders[key] = m.group(0)
    return key

content = WIKILINK.sub(_sub, content)

# 统计修复前
before = len(re.findall(r'\\\\[a-zA-Z]+', content))
print(f'Before: {before} double-backslash math commands')

# 将数学公式中的 \\X 替换为 \X
content = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', content)

# 恢复 wikilink
for k, v in placeholders.items():
    content = content.replace(k, v)

after = len(re.findall(r'\\\\[a-zA-Z]+', content))
print(f'After: {after} remaining (should be 0 except intentional)')

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fix complete')
