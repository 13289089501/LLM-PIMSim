# -*- coding: utf-8 -*-
"""v3 冒烟测试：验证 GUI 首页渲染 + API 端到端。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import gui_app
app = gui_app.app
c = app.test_client()

# 1. 首页渲染
r = c.get('/')
html = r.get_data(as_text=True)
print('index status:', r.status_code, 'len:', len(html))
for key in ['id="sel-experiment"', 'runSim', '/api/run', 'critical_path']:
    print('  contains', repr(key), ':', (key in html))

# 2. 实验列表
r = c.get('/api/experiments')
j = r.get_json()
print('experiments:', [e['name'] for e in j['experiments']])

# 3. 端到端 run（IC 参考）
payload = {'experiment': 'experiments/04_ic_reference.yaml', 'compute_map': {},
           'state': {}, 'run_validation': False}
r = c.post('/api/run', data=json.dumps(payload), content_type='application/json')
j = r.get_json()
print('run ok:', j.get('ok'), 'latency_ms=%.2f' % (j.get('total_latency_ms') or 0))
print('movement_total MB: %.1f' % ((j.get('movement_total_bytes') or 0) / 1e6))
print('critical_path ops:', len((j.get('critical_path') or {}).get('ops', [])))
print('SMOKE OK')
