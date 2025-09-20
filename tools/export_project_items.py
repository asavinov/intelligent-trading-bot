#!/usr/bin/env python3
"""
Export ProjectV2 items to CSV and JSON for easy verification and archival.

Usage:
  - Ensure GITHUB_TOKEN is set in the environment (or run via gh Actions where token is provided).
  - python tools/export_project_items.py --project-id PVT_kwHODCzAMM4BDOn_ --out-dir planning

This script uses the GraphQL API via PowerShell Invoke-RestMethod to avoid extra Python deps.
"""
import argparse
import json
import subprocess
import os
import csv
import tempfile
try:
    import requests
except Exception:
    requests = None
import urllib.request
import urllib.error


def run(cmd_list):
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print('ERROR:', res.stderr)
        raise SystemExit(res.returncode)
    return res.stdout


def gh_graphql_with_token(query, token=None):
    """Run the GraphQL query using the provided token or GITHUB_TOKEN env var.
    Falls back to gh CLI if no token is available.
    """
    if not token:
        token = os.environ.get('GITHUB_TOKEN')
    if not token:
        # try gh CLI but avoid quoting issues by writing payload to a temp file and using --input
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', encoding='utf-8') as tf:
                json.dump({'query': query}, tf, ensure_ascii=False)
                tfname = tf.name
            try:
                out = run(['gh', 'api', 'graphql', '--input', tfname])
                return out
            finally:
                try:
                    os.remove(tfname)
                except Exception:
                    pass
        except Exception:
            print('Need GITHUB_TOKEN or gh CLI available')
            raise SystemExit(1)

    # Prefer using requests (no shell quoting issues). If requests isn't available, fall back to PowerShell.
    # Prefer using requests (no shell quoting issues).
    if requests:
        headers = {
            'Authorization': f'bearer {token}',
            'Accept': 'application/json'
        }
        try:
            resp = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print('GraphQL request failed (network or HTTP error):', e)
            if hasattr(e, 'response') and e.response is not None:
                print('Response status:', e.response.status_code)
                print('Response body:', e.response.text)
            raise SystemExit(2)
    # fallback: use Python standard library (urllib) to POST the GraphQL query so we don't depend on PowerShell
    payload = json.dumps({'query': query}).encode('utf-8')
    req = urllib.request.Request(
        'https://api.github.com/graphql',
        data=payload,
        headers={
            'Authorization': f'bearer {token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = r.read().decode('utf-8')
            return out
    except urllib.error.HTTPError as he:
        try:
            body = he.read().decode('utf-8')
        except Exception:
            body = '<could not read error body>'
        print('GraphQL HTTP error:', he.code, he.reason)
        print('Response body:', body)
        raise SystemExit(3)
    except urllib.error.URLError as ue:
        print('GraphQL request failed (network error):', ue)
        raise SystemExit(4)


def export(project_id, out_dir, token=None):
    # Build a clear GraphQL query string and substitute the project node id safely.
    q_template = '''query {
  node(id: "PROJECT_ID") {
    ... on ProjectV2 {
      items(first: 500) {
        nodes {
          id
          content {
            __typename
            ... on Issue { number title url }
            ... on PullRequest { number title url }
          }
                    fieldValues(first: 50) {
                        nodes {
                            __typename
                            ... on ProjectV2ItemFieldSingleSelectValue {
                                optionId
                            }
                            ... on ProjectV2ItemFieldTextValue {
                                text
                            }
                            ... on ProjectV2ItemFieldNumberValue {
                                number
                            }
                        }
                    }
        }
      }
    }
  }
}'''
    q = q_template.replace('PROJECT_ID', project_id)

    # Debug: write the assembled query to a file so we can inspect syntax when gh reports parse errors
    try:
        with open(os.path.join('tools', '_last_query.graphql'), 'w', encoding='utf-8') as qq:
            qq.write(q)
    except Exception:
        pass

    out = gh_graphql_with_token(q, token=token)
    try:
        data = json.loads(out)
    except Exception as e:
        print('Failed to decode GraphQL response as JSON:', e)
        print('Raw response:\n', out)
        raise SystemExit(5)

    # If the GraphQL response doesn't include 'data', print any 'errors' payload for debugging
    if 'data' not in data:
        print('GraphQL response missing "data" key. Full response:')
        print(json.dumps(data, indent=2, ensure_ascii=False))
        # If there are GraphQL errors, show them
        if 'errors' in data:
            print('GraphQL errors:')
            print(json.dumps(data['errors'], indent=2, ensure_ascii=False))
        raise SystemExit(6)

    items = []
    nodes = data['data']['node']['items']['nodes']
    for n in nodes:
        item_id = n.get('id')
        content = n.get('content') or {}
        kind = content.get('__typename')
        number = content.get('number')
        title = content.get('title')
        url = content.get('url')

        # collect field values
        fv = {}
        for f in n.get('fieldValues', {}).get('nodes', []):
            field = f.get('projectField', {}).get('name')
            if f.get('singleSelectOption'):
                fv[field] = f['singleSelectOption'].get('name')
            elif f.get('text'):
                fv[field] = f.get('text')

        items.append({
            'item_id': item_id,
            'kind': kind,
            'number': number,
            'title': title,
            'url': url,
            'fields': fv,
        })

    # ensure out_dir
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'project_items_snapshot.json')
    csv_path = os.path.join(out_dir, 'project_items_snapshot.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # write CSV with columns: item_id, kind, number, title, url, Status, Priority, Effort
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['item_id', 'kind', 'number', 'title', 'url', 'Status', 'Priority', 'Effort'])
        for it in items:
            fv = it.get('fields', {})
            writer.writerow([
                it.get('item_id'),
                it.get('kind'),
                it.get('number'),
                it.get('title'),
                it.get('url'),
                fv.get('Status', ''),
                fv.get('Priority', ''),
                fv.get('Effort', ''),
            ])

    print('Wrote', json_path, csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-id', required=True)
    ap.add_argument('--out-dir', default='planning')
    ap.add_argument('--token', help='GitHub token to use for the GraphQL request')
    args = ap.parse_args()
    export(args.project_id, args.out_dir, token=args.token)


if __name__ == '__main__':
    main()
