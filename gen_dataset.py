#!/usr/bin/env python3
"""
Synthetic transaction dataset generator.

Usage:
    python3 gen_dataset.py --target-gb 4.5 --out transactions.csv [--seed 42] [--chunk 500000]
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reference pools
# --------------------------------------------------------------------------- #
FLC_CODES = np.array([7, 9, 17, 18, 28, 29, 37, 38, 49, 56, 66, 70, 76, 77, 80, 90, 94, 95])  # IRS file-location codes
FLC_W = np.array([8, 6, 5, 4, 7, 6, 8, 6, 7, 5, 2, 6, 3, 5, 4, 5, 6, 4], float); FLC_W /= FLC_W.sum()
TAX_CLASS = np.array([2, 1, 3, 4, 5]); TAX_CLASS_W = np.array([.80, .06, .06, .04, .04])
DOC_CODES = np.array([10, 11, 12, 21, 22, 26, 27, 70, 72, 73]); DOC_W = np.array([.22, .22, .18, .10, .08, .06, .06, .04, .02, .02])

EMAIL_DOMAINS = np.array(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com",
                          "comcast.net", "att.net", "protonmail.com", "live.com", "msn.com", "verizon.net",
                          "sbcglobal.net", "me.com", "ymail.com", "mail.com"])
EMAIL_DOMAIN_W = np.array([42, 14, 9, 8, 6, 4, 3, 2, 2, 2, 2, 1.5, 1.5, 1, 1, 1], float); EMAIL_DOMAIN_W /= EMAIL_DOMAIN_W.sum()

FIRST = np.array("james john robert michael william david richard joseph thomas charles chris daniel matthew anthony "
                 "mark donald steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey "
                 "ryan jacob gary nick eric jon stephen larry justin scott brandon ben sam greg frank alex raymond "
                 "patrick jack dennis jerry tyler aaron jose adam nathan henry doug zach peter kyle ethan walter noah "
                 "mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy betty sandra "
                 "margaret ashley kim emily donna michelle carol amanda melissa deborah stephanie rebecca sharon laura "
                 "cynthia dorothy amy kathleen angela shirley brenda emma anna pamela nicole samantha katherine "
                 "christine helen debra rachel carolyn janet maria catherine heather diane olivia julie joyce victoria "
                 "ruth virginia lauren kelly christina joan evelyn judith andrea hannah megan cheryl jacqueline martha "
                 "madison teresa gloria sara janice ann kathryn abigail sophia frances jean alice judy isabella julia "
                 "grace amber denise danielle marilyn beverly charlotte natalie theresa diana brittany doris kayla".split())
LAST = np.array("smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez "
                "wilson anderson thomas taylor moore jackson martin lee perez thompson white harris sanchez clark "
                "ramirez lewis robinson walker young allen king wright scott torres nguyen hill flores green adams "
                "nelson baker hall rivera campbell mitchell carter roberts gomez phillips evans turner diaz parker "
                "cruz edwards collins reyes stewart morris morales murphy cook rogers gutierrez ortiz morgan cooper "
                "peterson bailey reed kelly howard ramos kim cox ward richardson watson brooks chavez wood james "
                "bennett gray mendoza ruiz hughes price alvarez castillo sanders patel myers long ross foster jimenez "
                "powell jenkins perry russell sullivan bell coleman butler henderson barnes gonzales fisher vasquez "
                "simmons romero jordan patterson alexander hamilton graham reynolds griffin wallace moreno west cole "
                "hayes bryant herrera gibson ellis tran medina aguilar stevens murray ford castro marshall owens "
                "harrison fernandez mcdonald woods washington kennedy wells vargas henry chen freeman webb tucker "
                "guzman burns crawford olson simpson porter hunter gordon mendez silva shaw snyder mason dixon".split())

NAMED_BANKS = [  # (ABA prefix region code 01-12 / 21-32 for thrifts, weight)
    ("Chase", 2, 14), ("Bank of America", 2, 12), ("Wells Fargo", 12, 11), ("Citi", 2, 6), ("US Bank", 9, 5),
    ("PNC", 4, 5), ("Truist", 5, 4), ("Capital One", 5, 4), ("TD Bank", 1, 3), ("Regions", 6, 3),
    ("Fifth Third", 4, 3), ("Huntington", 4, 2), ("KeyBank", 4, 2), ("Citizens", 1, 2), ("M&T", 2, 2),
    ("Navy FCU", 25, 3), ("USAA", 31, 3), ("Ally", 7, 2), ("BMO", 7, 2), ("First Horizon", 8, 1),
    ("Synovus", 6, 1), ("Zions", 12, 1), ("Comerica", 11, 1), ("Frost", 11, 1), ("Webster", 1, 1),
] + [(f"Community Bank {i:02d}", None, 0.35) for i in range(1, 41)] + [(f"Credit Union {i:02d}", None, 0.35) for i in range(1, 41)]
PREPAID_BRANDS = [("Green Dot (MetaBank)", 7, 5), ("NetSpend (Pathward)", 7, 4), ("Walmart MoneyCard", 7, 2),
                  ("Bluebird (AmEx)", 2, 1), ("Chime (Bancorp)", 3, 6), ("Cash App (Sutton)", 4, 5),
                  ("PayPal/Venmo (Bancorp)", 3, 4), ("Serve", 2, 1), ("TurboTax Refund Card", 7, 2), ("Current (Choice)", 10, 1)]
PREPAID_SHARE = 0.065  # ~6.5% of rows go to prepaid-card routing numbers

STATUS = np.array(["single", "married_fs", "hoh"]); STATUS_W = np.array([.52, .33, .15])

PRIVATE_FIRST_OCTETS = {0, 10, 127, 100, 169, 172, 192, 198, 203} | set(range(224, 256))

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def aba_check_digit(d8):
    """d8: (n,8) int array of first eight digits -> 9th check digit."""
    s = 3 * (d8[:, 0] + d8[:, 3] + d8[:, 6]) + 7 * (d8[:, 1] + d8[:, 4] + d8[:, 7]) + (d8[:, 2] + d8[:, 5])
    return (10 - s % 10) % 10

def aba_is_valid(rtn):
    d = np.array([int(c) for c in rtn])
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0

def make_routing_pool(rng, banks):
    names, weights, rtns = [], [], []
    for name, prefix, w in banks:
        if prefix is None:
            prefix = int(rng.choice(np.r_[1:13, 21:33]))
        n_rtn = 3 if w >= 3 else 1  # big banks have several routing numbers
        for _ in range(n_rtn):
            while True:
                d8 = np.concatenate([[prefix // 10, prefix % 10], rng.integers(0, 10, 6)])
                rtn = "".join(map(str, d8)) + str(aba_check_digit(d8[None, :])[0])
                if rtn not in rtns:
                    break
            names.append(name); weights.append(w / n_rtn); rtns.append(rtn)
    weights = np.array(weights, float); weights /= weights.sum()
    return np.array(rtns), np.array(names), weights

def hex_str(rng, n, width):
    """Vectorized random lowercase hex strings of given width (multiple of 8)."""
    parts = [np.char.mod("%08x", rng.integers(0, 2**32, n, dtype=np.uint64)) for _ in range(width // 8)]
    out = parts[0]
    for p in parts[1:]:
        out = np.char.add(out, p)
    return out

def make_uuid4(rng, n, upper=False):
    """Vectorized RFC-4122 version-4 UUID strings."""
    def hx(width):
        return np.char.mod("%0" + str(width) + "x", rng.integers(0, 16**width, n, dtype=np.int64))
    var = rng.choice(np.array(list("89ab")), n)
    out = hx(8) + "-" + hx(4) + "-4" + hx(3) + "-" + var + hx(3) + "-" + hx(12)
    return np.char.upper(out) if upper else out

def make_phones(rng, n):
    npa_a = rng.integers(2, 10, n); npa_b = rng.integers(0, 9, n); npa_c = rng.integers(0, 10, n)
    bad = (npa_b == 1) & (npa_c == 1); npa_c[bad] = 2                         # no N11
    nxx_a = rng.integers(2, 10, n); nxx_b = rng.integers(0, 10, n); nxx_c = rng.integers(0, 10, n)
    bad = (nxx_b == 1) & (nxx_c == 1); nxx_c[bad] = 2
    bad = (nxx_a == 5) & (nxx_b == 5) & (nxx_c == 5); nxx_a[bad] = 6            # avoid 555
    line = rng.integers(0, 10000, n)
    return (np.char.mod("%d", npa_a * 100 + npa_b * 10 + npa_c) + "-"
            + np.char.mod("%d", nxx_a * 100 + nxx_b * 10 + nxx_c) + "-" + np.char.mod("%04d", line))

def make_emails(rng, n):
    f = rng.choice(FIRST, n); l = rng.choice(LAST, n)
    style = rng.integers(0, 6, n)
    num = np.char.mod("%d", rng.integers(1, 2000, n))
    yr = np.char.mod("%d", rng.integers(55, 100, n))
    local = np.where(style == 0, f + "." + l,
            np.where(style == 1, f + l + num,
            np.where(style == 2, f[:].astype("U1") + l + num,
            np.where(style == 3, f + "_" + l + yr,
            np.where(style == 4, l + "." + f + num, f + yr)))))
    dom = rng.choice(EMAIL_DOMAINS, n, p=EMAIL_DOMAIN_W)
    return np.char.add(np.char.add(local, "@"), dom)

def make_networks(rng, n):
    """Return (ipv4, ipv6) coherent pairs."""
    o1 = rng.integers(1, 224, n)
    bad = np.isin(o1, list(PRIVATE_FIRST_OCTETS))
    while bad.any():
        o1[bad] = rng.integers(1, 224, bad.sum()); bad = np.isin(o1, list(PRIVATE_FIRST_OCTETS))
    o2, o3, o4 = rng.integers(0, 256, n), rng.integers(0, 256, n), rng.integers(1, 255, n)
    ipv4 = (np.char.mod("%d", o1) + "." + np.char.mod("%d", o2) + "." + np.char.mod("%d", o3) + "." + np.char.mod("%d", o4))
    # IPv6 derived from the same provider: global prefix keyed off the v4 /16, interface id random
    pfx = rng.choice(np.array(["2001", "2600", "2601", "2603", "2605", "2607", "260a", "2620", "2a02", "2a03", "2001:db8"]), n,
                     p=[.14, .18, .18, .10, .08, .10, .04, .06, .04, .04, .04])
    h2 = np.char.mod("%x", (o1.astype(np.int64) << 8) | o2)
    h3 = np.char.mod("%x", (o3.astype(np.int64) << 4) | (o4 & 0xF))
    iid = [np.char.mod("%x", rng.integers(0, 65536, n)) for _ in range(4)]
    ipv6 = (pfx + ":" + h2 + ":" + h3 + "::" + iid[0] + ":" + iid[1] + ":" + iid[2] + ":" + iid[3])
    ipv4, ipv6 = np.asarray(ipv4), np.asarray(ipv6)
    _, keep = np.unique(ipv4, return_index=True)   # one IPv6 per IPv4
    return ipv4[keep], ipv6[keep]

def make_device_ids(rng, n):
    kind = rng.choice(4, n, p=[.34, .30, .11, .25])  # 0 windows, 1 ios, 2 mac, 3 android
    out = np.empty(n, dtype="U40")
    m = kind == 0; out[m] = np.char.add(np.char.add("{", make_uuid4(rng, m.sum(), upper=True)), "}")
    m = kind == 1; out[m] = make_uuid4(rng, m.sum(), upper=True)
    m = kind == 2; out[m] = make_uuid4(rng, m.sum(), upper=False)
    m = kind == 3; out[m] = hex_str(rng, m.sum(), 16)
    return out

def make_account_numbers(rng, n):
    length = rng.choice(np.arange(8, 13), n, p=[.14, .22, .34, .18, .12])
    val = rng.integers(0, 10**12, n, dtype=np.int64)
    s = np.char.mod("%012d", val)
    return np.array([x[-k:] for x, k in zip(s, length)])  # right-trim to requested length

# --------------------------------------------------------------------------- #
# Submission-date model: late Jan -> Oct 15 of the year after the reporting year,
# peaking around mid-April with a second bump at the October extension deadline.
# --------------------------------------------------------------------------- #
def window_bounds(reporting_year, today):
    y = reporting_year + 1
    start = pd.Timestamp(f"{y}-01-27")
    end = pd.Timestamp(f"{y}-10-15 23:59:59")
    return start, min(end, today)

def sample_dates(rng, n, reporting_year, today):
    start, end = window_bounds(reporting_year, today)
    total_days = (end - start).days + 1
    apr15 = (pd.Timestamp(f"{reporting_year + 1}-04-15") - start).days
    oct15 = (pd.Timestamp(f"{reporting_year + 1}-10-15") - start).days
    comp = rng.choice(3, n, p=[.72, .18, .10])
    d = np.empty(n)
    m = comp == 0; d[m] = rng.normal(apr15 - 14, 22, m.sum())            # main season
    m = comp == 1; d[m] = rng.uniform(0, total_days, m.sum())             # background
    m = comp == 2; d[m] = oct15 - rng.exponential(9, m.sum())             # extension rush
    d = np.clip(d, 0, total_days - 1e-6)
    # time of day: weighted toward business/evening hours
    hw = np.array([1,1,.5,.5,.5,1,2,4,6,8,9,9,9,9,9,9,9,8,8,8,7,6,4,2], float)
    hour = rng.choice(24, n, p=hw / hw.sum())
    sec = hour * 3600 + rng.integers(0, 3600, n)
    ts = start.value + (np.floor(d).astype(np.int64) * 86400 + sec) * 10**9
    return pd.to_datetime(ts)

# --------------------------------------------------------------------------- #
def build_chunk(rng, n, pools, today):
    ry = rng.choice([2024, 2025], n, p=[.47, .53])
    ts_arr = np.empty(n, dtype="datetime64[ns]")
    for y in (2024, 2025):
        m = ry == y
        ts_arr[m] = sample_dates(rng, m.sum(), y, today).values
    ts = pd.DatetimeIndex(ts_arr)

    # DocNum = FLC(2) tax class(1) doc code(2) julian(3) block(3) serial(2) year digit(1)
    flc = rng.choice(FLC_CODES, n, p=FLC_W)
    tc = rng.choice(TAX_CLASS, n, p=TAX_CLASS_W)
    dc = rng.choice(DOC_CODES, n, p=DOC_W)
    jd = ts.dayofyear.values.astype(np.int64)
    blk = rng.integers(0, 1000, n); ser = rng.integers(0, 100, n); yd = ts.year.values % 10
    docnum_int = (flc * 10**12 + tc * 10**11 + dc * 10**9 + jd * 10**6 + blk * 10**3 + ser * 10 + yd).astype(np.int64)
    docnum = np.char.mod("%014d", docnum_int)

    # identity pools (sampled with reuse so emails/devices/networks recur realistically)
    ei = rng.integers(0, len(pools["email"]), n); pi = rng.integers(0, len(pools["phone"]), n)
    ni = rng.integers(0, len(pools["ipv4"]), n); di = rng.integers(0, len(pools["device"]), n)
    ai = rng.integers(0, len(pools["acct"]), n)
    prepaid = rng.random(n) < PREPAID_SHARE
    rtn = np.where(prepaid, rng.choice(pools["prepaid_rtn"], n, p=pools["prepaid_w"]),
                   rng.choice(pools["bank_rtn"], n, p=pools["bank_w"]))

    status = rng.choice(STATUS, n, p=STATUS_W)
    trans_amt = np.round(np.exp(rng.normal(7.4, 0.9, n)), 2)                # median ~$1,640
    trans_amt = np.clip(trans_amt, 25, 85000)
    def disc(p, lo, hi):
        has = rng.random(n) < p
        return np.round(np.where(has, trans_amt * rng.uniform(lo, hi, n), 0.0), 2)
    d1 = disc(.45, .02, .12); d2 = disc(.22, .01, .08); d3 = disc(.08, .005, .05)
    adj = np.round(np.maximum(trans_amt - d1 - d2 - d3, 0.0), 2)
    refund = np.round(np.where(rng.random(n) < .11, adj * rng.uniform(.05, 1.0, n), 0.0), 2)

    return pd.DataFrame({
        "DocNum": docnum,
        "UUID": make_uuid4(rng, n),
        "ReportingYear": ry,
        "SubmissionDate": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "RoutingNumber": rtn,
        "AccountNumber": pools["acct"][ai],
        "Email": pools["email"][ei],
        "Phone": pools["phone"][pi],
        "IPv4": pools["ipv4"][ni],
        "IPv6": pools["ipv6"][ni],
        "DeviceID": pools["device"][di],
        "trans_amt": trans_amt,
        "adj_trans_amt": adj,
        "refund_amt": refund,
        "customer_status": status,
        "discount_1": d1, "discount_2": d2, "discount_3": d3,
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=4.5)
    ap.add_argument("--out", default="transactions.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--rows", type=int, default=None, help="override: exact row count")
    ap.add_argument("--pool-scale", type=float, default=1.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    today = pd.Timestamp("2026-10-06 23:59:59")

    t0 = time.time()
    bank_rtn, bank_names, bank_w = make_routing_pool(rng, NAMED_BANKS)
    pre_rtn, pre_names, pre_w = make_routing_pool(rng, PREPAID_BRANDS)
    assert all(aba_is_valid(r) for r in np.r_[bank_rtn, pre_rtn])
    pd.DataFrame({"RoutingNumber": np.r_[bank_rtn, pre_rtn], "Institution": np.r_[bank_names, pre_names],
                  "Type": ["bank"] * len(bank_rtn) + ["prepaid_card"] * len(pre_rtn),
                  "Weight": np.r_[bank_w * (1 - PREPAID_SHARE), pre_w * PREPAID_SHARE]}
                 ).to_csv(os.path.splitext(args.out)[0] + "_routing_lookup.csv", index=False)

    s = args.pool_scale
    ipv4, ipv6 = make_networks(rng, int(900_000 * s))
    pools = dict(bank_rtn=bank_rtn, bank_w=bank_w, prepaid_rtn=pre_rtn, prepaid_w=pre_w,
                 email=make_emails(rng, int(2_200_000 * s)), phone=make_phones(rng, int(2_000_000 * s)),
                 ipv4=ipv4, ipv6=ipv6, device=make_device_ids(rng, int(1_800_000 * s)),
                 acct=make_account_numbers(rng, int(2_000_000 * s)))
    print(f"pools ready in {time.time()-t0:.1f}s", flush=True)

    target = args.target_gb * 1024**3
    written = 0; rows = 0; first = True
    with open(args.out, "w", newline="") as fh:
        while True:
            if args.rows is not None:
                n = min(args.chunk, args.rows - rows)
                if n <= 0: break
            else:
                n = args.chunk
                if written and rows:
                    remaining = target - written
                    n = int(min(args.chunk, max(0, remaining / (written / rows))))
                    if n <= 0: break
            df = build_chunk(rng, n, pools, today)
            df.to_csv(fh, index=False, header=first, lineterminator="\n")
            first = False
            written = fh.tell(); rows += n
            print(f"rows={rows:,}  size={written/1024**3:.2f} GB  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done: {rows:,} rows, {written/1024**3:.2f} GB -> {args.out}")

if __name__ == "__main__":
    main()
