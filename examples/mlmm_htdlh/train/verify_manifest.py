
import json, hashlib, sys
root=sys.argv[1]; m=json.load(open(root+"/manifest.json")); bad=0
for rel,h in m["files"].items():
    hh=hashlib.sha256()
    with open(f"{root}/{rel}","rb") as f:
        for b in iter(lambda: f.read(1<<22), b""): hh.update(b)
    if hh.hexdigest()!=h: print("MISMATCH",rel); bad+=1
print("verified",len(m["files"]),"files, mismatches",bad); sys.exit(1 if bad else 0)
