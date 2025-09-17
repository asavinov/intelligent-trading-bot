import urllib.request, os
job='f5fb3469-6d46-4a39-9d8f-2e7c83f60b4a'
base='http://127.0.0.1:8000'
endpoints=[f'/api/scripts/debug/job/{job}', '/api/scripts/debug/active_processes']
for ep in endpoints:
    url=base+ep
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print('\n==',url,'status',r.status)
            data=r.read().decode('utf-8',errors='ignore')
            print(data)
    except Exception as e:
        print('\n==',url,'ERROR',e)
# print log files
so=f'logs/jobs/{job}.stdout.log'
se=f'logs/jobs/{job}.stderr.log'
print('\n== stdout file:', so, 'exists=', os.path.exists(so))
if os.path.exists(so):
    with open(so,'r',encoding='utf-8',errors='ignore') as f:
        s=f.read()
        print('\n-- stdout content (last 2000 chars) --')
        print(s[-2000:])
print('\n== stderr file:', se, 'exists=', os.path.exists(se))
if os.path.exists(se):
    with open(se,'r',encoding='utf-8',errors='ignore') as f:
        s=f.read()
        print('\n-- stderr content (last 2000 chars) --')
        print(s[-2000:])
