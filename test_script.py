import sys, json
sys.stdout.reconfigure(encoding='utf-8')
json.dump([1,2,3], sys.stdout)
