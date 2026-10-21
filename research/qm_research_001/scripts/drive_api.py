#!/usr/bin/env python3
"""Drive helper: read rclone token from config, refresh if needed, call Drive API.
Never prints secrets."""
import json, os, re, sys, time, urllib.parse, urllib.request

CONF = os.path.expanduser("~/.config/rclone/rclone.conf")

def load_conf():
    txt = open(CONF).read()
    sec = txt.split("[gdrive]", 1)[1]
    out = {}
    for line in sec.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def token():
    c = load_conf()
    t = json.loads(c["token"])
    if t.get("expiry"):
        exp = time.mktime(time.strptime(t["expiry"][:19], "%Y-%m-%dT%H:%M:%S"))
        if exp < time.time() + 60:
            data = urllib.parse.urlencode({
                "client_id": c.get("client_id", ""),
                "client_secret": c.get("client_secret", ""),
                "refresh_token": t["refresh_token"],
                "grant_type": "refresh_token",
            }).encode()
            req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
            r = json.load(urllib.request.urlopen(req))
            t["access_token"] = r["access_token"]
            t["expiry"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime(time.time() + r.get("expires_in", 3600)))
    return t["access_token"]

def api(path, params=None, raw=False):
    url = "https://www.googleapis.com/drive/v3/" + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token()})
    try:
        r = urllib.request.urlopen(req)
        return r.read() if raw else json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %s: %s" % (e.code, e.read()[:500]))

def export(file_id, mime, outpath):
    data = api("files/%s/export" % file_id, {"mimeType": mime}, raw=True)
    open(outpath, "wb").write(data)
    return len(data)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "meta":
        print(json.dumps(api("files/%s" % sys.argv[2], {"fields": "id,name,mimeType,parents,modifiedTime,size"}), ensure_ascii=False, indent=1))
    elif cmd == "list":
        print(json.dumps(api("files", {"q": "'%s' in parents and trashed=false" % sys.argv[2],
                                       "fields": "files(id,name,mimeType,size,modifiedTime)", "pageSize": 200,
                                       "orderBy": "name"}), ensure_ascii=False, indent=1))
    elif cmd == "export":
        print(export(sys.argv[2], sys.argv[3], sys.argv[4]))
