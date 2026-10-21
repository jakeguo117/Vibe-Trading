#!/usr/bin/env python3
"""Crawl SEC EDGAR 8-K Item 2.02 filings (earnings releases) for QM symbols.

Point-in-time fields kept: acceptanceDateTime (UTC), filingDate, accession, items.
Availability rule used downstream: the filing is visible only after its EDGAR
acceptance timestamp; the 8-K is treated as a conservative proxy for the
earnings press release (never later than the release).
"""
from __future__ import annotations

import json, os, sys, time, urllib.request, urllib.error

UA = "QM-RESEARCH-001 research contact jakeguo117@gmail.com"
BASE = "https://data.sec.gov"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    data = gzip.decompress(data)
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(1.5 * (i + 1)); continue
            if e.code == 404:
                return None
            time.sleep(0.5 * (i + 1))
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None


def rows_from(recent, cik):
    out = []
    n = len(recent.get("form", []))
    for i in range(n):
        if recent["form"][i] != "8-K":
            continue
        items = recent.get("items", [""] * n)[i] or ""
        if "2.02" not in items:
            continue
        out.append({
            "cik": cik, "filingDate": recent["filingDate"][i],
            "acceptance": recent["acceptanceDateTime"][i],
            "items": items, "accn": recent["accessionNumber"][i],
        })
    return out


def main(sym_file, out_csv, limit=None):
    tickers = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    syms = [s.strip() for s in open(sym_file) if s.strip()]
    if limit:
        syms = syms[:limit]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            next(f, None)
            for line in f:
                done.add(line.split(",")[0])
    rows = []
    t0 = time.time()
    for i, s in enumerate(syms):
        if s in done:
            continue
        cik = tickers.get(s)
        if not cik:
            rows.append({"sym": s, "cik": "", "filingDate": "", "acceptance": "", "items": "", "accn": "NO_CIK"})
            continue
        d = get(f"{BASE}/submissions/CIK{int(cik):010d}.json")
        if not d:
            rows.append({"sym": s, "cik": cik, "filingDate": "", "acceptance": "", "items": "", "accn": "FETCH_FAIL"})
            continue
        rec = rows_from(d["filings"]["recent"], cik)
        for extra in d["filings"].get("files", []):
            name = extra.get("name")
            if not name:
                continue
            if extra.get("filingTo", "9999") < "2015":
                continue
            e = get(f"{BASE}/submissions/{name}")
            if e:
                rec += rows_from(e, cik)
        for r in rec:
            r["sym"] = s
            rows.append(r)
        if i % 100 == 0:
            print(f"{i}/{len(syms)} {s} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)
        time.sleep(0.12)
    import csv
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sym", "cik", "filingDate", "acceptance", "items", "accn"])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print("done rows", len(rows), "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], None if len(sys.argv) < 5 else int(sys.argv[4]))
