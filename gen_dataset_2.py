#!/usr/bin/env python3
"""
Synthetic transaction dataset generator
=======================================

Produces a large CSV (default ~4.5 GB, ~18.7M rows) of fake tax-style
transactions, then optionally splits it into 25 MB parts.  This script is the
plain-Python twin of ``gen_dataset.ipynb`` -- same sections, same defaults,
same output for the same seed.

Columns produced
----------------
DocNum          14-digit DLN-style string:
                FLC(2) + tax class(1) + doc code(2) + Julian day(3) + block(3)
                + serial(2) + year digit(1).  The Julian day and the year digit
                are computed FROM SubmissionDate, so they always agree with it.
UUID            RFC-4122 version-4 UUID, unique per row.
ReportingYear   2024 or 2025.
SubmissionDate  Timestamp inside that reporting year's filing window:
                Jan 27 of the following year through Oct 15 of the following
                year (capped at TODAY so no row is dated in the future).
RoutingNumber   9-digit string that passes the ABA check-digit test.  Drawn
                from a weighted pool of "banks" (~93.5%) plus a small pool of
                prepaid-card issuers (~6.5%).
AccountNumber   8-12 digit string.
Email, Phone    Plausible email; phone is NANP-valid in ###-###-#### form.
IPv4, IPv6      Public IPv4 and a matching IPv6.  A "network" is an (IPv4,
                IPv6) pair, so the same IPv4 always appears with the same IPv6.
DeviceID        Mix of GUID-style IDs ({...} Windows, UPPER iOS, lower Mac)
                and 16-hex Android IDs.
trans_amt       Gross transaction amount (lognormal, median ~$1,600).
discount_1..3   Optional discounts taken off the gross amount.
adj_trans_amt   trans_amt minus the discounts (never below zero).
refund_amt      Non-zero in ~11% of rows; a fraction of adj_trans_amt.
customer_status single / married_fs / hoh.

How it works
------------
1. Build "identity pools" (emails, phones, accounts, devices, networks) once.
   Rows then SAMPLE from these pools, so the same customer identifiers recur
   across many rows the way real data does, instead of every row being unique.
2. Generate rows in chunks of CHUNK (default 500k) using vectorised numpy so
   the whole file is produced in a few minutes, and append each chunk to the
   CSV.  Stop when the file reaches TARGET_GB (or after ROWS rows).
3. Read the first 200k rows back and verify every column rule.
4. Optionally split the big CSV into line-aligned parts of <= CHUNK_MB each.

Requires only numpy and pandas.
"""

# ============================================================================ #
# Configuration                                                                 #
# ============================================================================ #
# Edit these, or override them on the command line (see --help).

TARGET_GB  = 4.5                 # approximate output size in GB (ignored if ROWS is set)
ROWS       = None                # e.g. 200_000 for a quick test; None = fill to TARGET_GB
OUT_PATH   = "transactions.csv"  # output CSV; a *_routing_lookup.csv is written next to it
SEED       = 42                  # same seed -> identical file every run
CHUNK      = 500_000             # rows generated per write (lower this if memory is tight)
POOL_SCALE = 1.0                 # multiplier on the size of the identity pools (see below)
TODAY      = "2026-10-06"        # submission dates are capped here (no future-dated rows)

# Splitting (step 4).  Set SPLIT = False to skip it.
SPLIT               = True
CHUNK_MB            = 25         # max size of each part, in MB
PARTS_DIR           = None       # None -> "<OUT_PATH stem>_parts/" next to the CSV
HEADER_IN_EACH_PART = True       # True: every part is a standalone CSV with a header row

import argparse
import os
import time
import uuid

import numpy as np
import pandas as pd


# ============================================================================ #
# Reference pools                                                               #
# ============================================================================ #
# These are the "vocabularies" the generator draws from.  Each has a matching
# weight array so common values appear more often than rare ones.

# --- DocNum building blocks --------------------------------------------------
# File Location Codes: the two leading digits of an IRS DLN identify the
# campus / service center that processed the document.  These are real codes.
FLC_CODES = np.array([7, 9, 17, 18, 28, 29, 37, 38, 49, 56, 66, 70, 76, 77, 80, 90, 94, 95])
FLC_W = np.array([8, 6, 5, 4, 7, 6, 8, 6, 7, 5, 2, 6, 3, 5, 4, 5, 6, 4], float)
FLC_W /= FLC_W.sum()                                   # normalise to probabilities

# Tax class digit.  2 = individual income tax, which dominates; others are rare.
TAX_CLASS = np.array([2, 1, 3, 4, 5])
TAX_CLASS_W = np.array([.80, .06, .06, .04, .04])

# Two-digit document codes (which form / transaction type was filed).
DOC_CODES = np.array([10, 11, 12, 21, 22, 26, 27, 70, 72, 73])
DOC_W = np.array([.22, .22, .18, .10, .08, .06, .06, .04, .02, .02])

# --- Email building blocks ---------------------------------------------------
EMAIL_DOMAINS = np.array(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com",
                          "comcast.net", "att.net", "protonmail.com", "live.com", "msn.com", "verizon.net",
                          "sbcglobal.net", "me.com", "ymail.com", "mail.com"])
EMAIL_DOMAIN_W = np.array([42, 14, 9, 8, 6, 4, 3, 2, 2, 2, 2, 1.5, 1.5, 1, 1, 1], float)
EMAIL_DOMAIN_W /= EMAIL_DOMAIN_W.sum()

# Common US first and last names; emails are assembled from these in a few
# styles (first.last, firstlast123, f_last88, ...).
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

# --- Financial institutions --------------------------------------------------
# Each entry is (name, ABA prefix, weight).
#   * The ABA prefix is the first two digits of a routing number: 01-12 are the
#     Federal Reserve districts, 21-32 are the same districts for thrifts/credit
#     unions.  None means "pick a random valid prefix".
#   * The weight controls how often that institution's routing numbers appear.
#   * Big banks (weight >= 3) get three routing numbers each; small ones get one.
# The routing numbers themselves are generated (not real), but every one
# passes the ABA check-digit test.
NAMED_BANKS = [
    ("Chase", 2, 14), ("Bank of America", 2, 12), ("Wells Fargo", 12, 11), ("Citi", 2, 6), ("US Bank", 9, 5),
    ("PNC", 4, 5), ("Truist", 5, 4), ("Capital One", 5, 4), ("TD Bank", 1, 3), ("Regions", 6, 3),
    ("Fifth Third", 4, 3), ("Huntington", 4, 2), ("KeyBank", 4, 2), ("Citizens", 1, 2), ("M&T", 2, 2),
    ("Navy FCU", 25, 3), ("USAA", 31, 3), ("Ally", 7, 2), ("BMO", 7, 2), ("First Horizon", 8, 1),
    ("Synovus", 6, 1), ("Zions", 12, 1), ("Comerica", 11, 1), ("Frost", 11, 1), ("Webster", 1, 1),
] + [(f"Community Bank {i:02d}", None, 0.35) for i in range(1, 41)] \
  + [(f"Credit Union {i:02d}", None, 0.35) for i in range(1, 41)]

# Prepaid-card programs.  Their routing numbers belong to the sponsor bank
# behind the card, which is why e.g. Green Dot and NetSpend share prefix 07.
PREPAID_BRANDS = [("Green Dot (MetaBank)", 7, 5), ("NetSpend (Pathward)", 7, 4), ("Walmart MoneyCard", 7, 2),
                  ("Bluebird (AmEx)", 2, 1), ("Chime (Bancorp)", 3, 6), ("Cash App (Sutton)", 4, 5),
                  ("PayPal/Venmo (Bancorp)", 3, 4), ("Serve", 2, 1), ("TurboTax Refund Card", 7, 2), ("Current (Choice)", 10, 1)]
PREPAID_SHARE = 0.065  # fraction of rows whose RoutingNumber is a prepaid-card issuer

# --- Customer status ---------------------------------------------------------
STATUS = np.array(["single", "married_fs", "hoh"])
STATUS_W = np.array([.52, .33, .15])

# --- IPv4 ranges to avoid ----------------------------------------------------
# First octets that are private, loopback, link-local, documentation, or
# multicast/reserved.  We want every IPv4 to look like a public address.
PRIVATE_FIRST_OCTETS = {0, 10, 127, 100, 169, 172, 192, 198, 203} | set(range(224, 256))


# ============================================================================ #
# Field generators                                                              #
# ============================================================================ #
# Everything here is vectorised: each function takes a numpy Generator `rng`
# and a count `n`, and returns an array of n values.  Avoiding Python loops is
# what keeps the 18M-row build down to a few minutes.

def aba_check_digit(d8):
    """Compute the 9th (check) digit of a routing number from its first eight.

    `d8` is an (n, 8) integer array.  The ABA rule: weight the digits
    3,7,1,3,7,1,3,7,(1) and the total must be a multiple of 10.
    """
    s = 3 * (d8[:, 0] + d8[:, 3] + d8[:, 6]) + 7 * (d8[:, 1] + d8[:, 4] + d8[:, 7]) + (d8[:, 2] + d8[:, 5])
    return (10 - s % 10) % 10


def aba_is_valid(rtn):
    """True if the 9-digit routing-number string passes the ABA checksum."""
    d = np.array([int(c) for c in rtn])
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


def make_routing_pool(rng, banks):
    """Turn a list of (name, prefix, weight) into parallel arrays of
    routing numbers, institution names, and normalised sampling weights.

    A bank with weight >= 3 gets three distinct routing numbers (large banks
    really do have many); its weight is shared equally among them.
    """
    names, weights, rtns = [], [], []
    for name, prefix, w in banks:
        if prefix is None:                                   # small institution: random district
            prefix = int(rng.choice(np.r_[1:13, 21:33]))
        n_rtn = 3 if w >= 3 else 1
        for _ in range(n_rtn):
            while True:                                      # loop until we get an unused number
                d8 = np.concatenate([[prefix // 10, prefix % 10], rng.integers(0, 10, 6)])
                rtn = "".join(map(str, d8)) + str(aba_check_digit(d8[None, :])[0])
                if rtn not in rtns:
                    break
            names.append(name); weights.append(w / n_rtn); rtns.append(rtn)
    weights = np.array(weights, float); weights /= weights.sum()
    return np.array(rtns), np.array(names), weights


def hex_str(rng, n, width):
    """n random lowercase hex strings of `width` chars (width must be a multiple of 8).
    Built 8 hex chars (32 bits) at a time and concatenated."""
    parts = [np.char.mod("%08x", rng.integers(0, 2**32, n, dtype=np.uint64)) for _ in range(width // 8)]
    out = parts[0]
    for p in parts[1:]:
        out = np.char.add(out, p)
    return out


def make_uuid4(rng, n, upper=False):
    """n RFC-4122 version-4 UUID strings, e.g. 8ca5fe05-d593-42f4-88c3-2474c49be7eb.

    Layout is 8-4-4-4-12 hex digits.  The third group must start with '4'
    (version) and the fourth group with 8, 9, a or b (variant); the other
    30 hex digits are random.
    """
    def hx(width):
        return np.char.mod("%0" + str(width) + "x", rng.integers(0, 16**width, n, dtype=np.int64))
    var = rng.choice(np.array(list("89ab")), n)
    out = hx(8) + "-" + hx(4) + "-4" + hx(3) + "-" + var + hx(3) + "-" + hx(12)
    return np.char.upper(out) if upper else out


def make_phones(rng, n):
    """n NANP-valid phone numbers as ###-###-####.

    NANP rules enforced:
      * area code (NPA):  first digit 2-9, second digit 0-8, not N11 (e.g. 911)
      * exchange (NXX):   first digit 2-9, not N11, and we also avoid 555
      * line number:      any 4 digits
    """
    npa_a = rng.integers(2, 10, n); npa_b = rng.integers(0, 9, n); npa_c = rng.integers(0, 10, n)
    bad = (npa_b == 1) & (npa_c == 1); npa_c[bad] = 2                         # no N11 area codes
    nxx_a = rng.integers(2, 10, n); nxx_b = rng.integers(0, 10, n); nxx_c = rng.integers(0, 10, n)
    bad = (nxx_b == 1) & (nxx_c == 1); nxx_c[bad] = 2                         # no N11 exchanges
    bad = (nxx_a == 5) & (nxx_b == 5) & (nxx_c == 5); nxx_a[bad] = 6            # no 555 exchange
    line = rng.integers(0, 10000, n)
    return (np.char.mod("%d", npa_a * 100 + npa_b * 10 + npa_c) + "-"
            + np.char.mod("%d", nxx_a * 100 + nxx_b * 10 + nxx_c) + "-" + np.char.mod("%04d", line))


def make_emails(rng, n):
    """n email addresses assembled from the name lists in one of six styles."""
    f = rng.choice(FIRST, n); l = rng.choice(LAST, n)
    style = rng.integers(0, 6, n)
    num = np.char.mod("%d", rng.integers(1, 2000, n))       # e.g. johnsmith1234
    yr = np.char.mod("%d", rng.integers(55, 100, n))        # e.g. john_smith88 (birth-year style)
    local = np.where(style == 0, f + "." + l,
            np.where(style == 1, f + l + num,
            np.where(style == 2, f[:].astype("U1") + l + num,   # first initial + last + number
            np.where(style == 3, f + "_" + l + yr,
            np.where(style == 4, l + "." + f + num, f + yr)))))
    dom = rng.choice(EMAIL_DOMAINS, n, p=EMAIL_DOMAIN_W)
    return np.char.add(np.char.add(local, "@"), dom)


def make_networks(rng, n):
    """Return coherent (ipv4, ipv6) arrays: one unique public IPv4 and the
    IPv6 that always accompanies it.

    The IPv6 is derived from the IPv4's octets (a common ISP prefix, then the
    v4 /16 and /24 encoded as hextets, then a random interface id), so the two
    look like they come from the same provider.
    """
    # IPv4: resample any first octet that falls in a private/reserved range
    o1 = rng.integers(1, 224, n)
    bad = np.isin(o1, list(PRIVATE_FIRST_OCTETS))
    while bad.any():
        o1[bad] = rng.integers(1, 224, bad.sum()); bad = np.isin(o1, list(PRIVATE_FIRST_OCTETS))
    o2, o3, o4 = rng.integers(0, 256, n), rng.integers(0, 256, n), rng.integers(1, 255, n)
    ipv4 = (np.char.mod("%d", o1) + "." + np.char.mod("%d", o2) + "." + np.char.mod("%d", o3) + "." + np.char.mod("%d", o4))

    # IPv6: <ISP prefix>:<hextet from o1,o2>:<hextet from o3,o4>::<4 random hextets>
    pfx = rng.choice(np.array(["2001", "2600", "2601", "2603", "2605", "2607", "260a", "2620", "2a02", "2a03", "2001:db8"]), n,
                     p=[.14, .18, .18, .10, .08, .10, .04, .06, .04, .04, .04])
    h2 = np.char.mod("%x", (o1.astype(np.int64) << 8) | o2)
    h3 = np.char.mod("%x", (o3.astype(np.int64) << 4) | (o4 & 0xF))
    iid = [np.char.mod("%x", rng.integers(0, 65536, n)) for _ in range(4)]
    ipv6 = (pfx + ":" + h2 + ":" + h3 + "::" + iid[0] + ":" + iid[1] + ":" + iid[2] + ":" + iid[3])

    # Deduplicate on IPv4 so every IPv4 maps to exactly one IPv6
    ipv4, ipv6 = np.asarray(ipv4), np.asarray(ipv6)
    _, keep = np.unique(ipv4, return_index=True)
    return ipv4[keep], ipv6[keep]


def make_device_ids(rng, n):
    """n device IDs in a platform mix:
         34% Windows  -> GUID in braces, uppercase   {46CCB5A7-8CE7-42BF-AD9C-CE05CA1E80B2}
         30% iOS      -> UUID uppercase              D71B2918-7908-4121-9C55-60152EB6FC21
         11% Mac      -> UUID lowercase              b3b22c9f-8cad-45de-970b-ae821f6ead8b
         25% Android  -> 16 hex chars (ANDROID_ID)   b71cbb3c94d37093
    """
    kind = rng.choice(4, n, p=[.34, .30, .11, .25])
    out = np.empty(n, dtype="U40")
    m = kind == 0; out[m] = np.char.add(np.char.add("{", make_uuid4(rng, m.sum(), upper=True)), "}")
    m = kind == 1; out[m] = make_uuid4(rng, m.sum(), upper=True)
    m = kind == 2; out[m] = make_uuid4(rng, m.sum(), upper=False)
    m = kind == 3; out[m] = hex_str(rng, m.sum(), 16)
    return out


def make_account_numbers(rng, n):
    """n account numbers of 8-12 digits (10 digits most common).
    A 12-digit zero-padded number is generated and right-trimmed to the
    chosen length, so leading zeros are possible like in real account numbers."""
    length = rng.choice(np.arange(8, 13), n, p=[.14, .22, .34, .18, .12])
    val = rng.integers(0, 10**12, n, dtype=np.int64)
    s = np.char.mod("%012d", val)
    return np.array([x[-k:] for x, k in zip(s, length)])


# ============================================================================ #
# Submission-date model                                                         #
# ============================================================================ #
# Reporting year Y is filed during Y+1.  The window opens in late January and
# closes at the October 15 extension deadline.  Dates are drawn from a mixture:
#   72%  main season   - normal curve peaking ~2 weeks before April 15
#   18%  background    - uniform over the whole window
#   10%  extension     - exponential pile-up just before October 15
# Times of day are weighted toward daytime and evening hours.

def window_bounds(reporting_year, today):
    """(start, end) timestamps of the filing window for a reporting year,
    with the end capped at `today` so nothing is dated in the future."""
    y = reporting_year + 1
    start = pd.Timestamp(f"{y}-01-27")
    end = pd.Timestamp(f"{y}-10-15 23:59:59")
    return start, min(end, today)


def sample_dates(rng, n, reporting_year, today):
    """n timestamps inside the reporting year's window, as a DatetimeIndex."""
    start, end = window_bounds(reporting_year, today)
    total_days = (end - start).days + 1
    apr15 = (pd.Timestamp(f"{reporting_year + 1}-04-15") - start).days   # day offset of April 15
    oct15 = (pd.Timestamp(f"{reporting_year + 1}-10-15") - start).days   # day offset of October 15

    comp = rng.choice(3, n, p=[.72, .18, .10])                            # which mixture component
    d = np.empty(n)                                                        # day offset from `start`
    m = comp == 0; d[m] = rng.normal(apr15 - 14, 22, m.sum())             # main season
    m = comp == 1; d[m] = rng.uniform(0, total_days, m.sum())              # background
    m = comp == 2; d[m] = oct15 - rng.exponential(9, m.sum())              # extension rush
    d = np.clip(d, 0, total_days - 1e-6)                                   # keep inside the window

    # Hour-of-day weights (index = hour 0..23): quiet overnight, busy 9am-9pm
    hw = np.array([1, 1, .5, .5, .5, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 7, 6, 4, 2], float)
    hour = rng.choice(24, n, p=hw / hw.sum())
    sec = hour * 3600 + rng.integers(0, 3600, n)                           # random minute/second within the hour

    ts = start.value + (np.floor(d).astype(np.int64) * 86400 + sec) * 10**9   # nanoseconds since epoch
    return pd.to_datetime(ts)


# ============================================================================ #
# Row builder                                                                   #
# ============================================================================ #

def build_chunk(rng, n, pools, today):
    """Build one DataFrame of n rows.  `pools` holds the pre-built identity
    pools and routing-number pools (see run_generation)."""

    # --- reporting year and submission timestamp ---------------------------
    ry = rng.choice([2024, 2025], n, p=[.47, .53])
    ts_arr = np.empty(n, dtype="datetime64[ns]")
    for y in (2024, 2025):                       # each year has its own window
        m = ry == y
        ts_arr[m] = sample_dates(rng, m.sum(), y, today).values
    ts = pd.DatetimeIndex(ts_arr)

    # --- DocNum ------------------------------------------------------------
    # FLC(2) tax class(1) doc code(2) Julian day(3) block(3) serial(2) year digit(1)
    # Julian day and year digit come from the timestamp so they always agree.
    flc = rng.choice(FLC_CODES, n, p=FLC_W)
    tc = rng.choice(TAX_CLASS, n, p=TAX_CLASS_W)
    dc = rng.choice(DOC_CODES, n, p=DOC_W)
    jd = ts.dayofyear.values.astype(np.int64)          # 001-366
    blk = rng.integers(0, 1000, n)                     # block number 000-999
    ser = rng.integers(0, 100, n)                      # serial 00-99
    yd = ts.year.values % 10                           # last digit of the processing year
    docnum_int = (flc * 10**12 + tc * 10**11 + dc * 10**9 + jd * 10**6 + blk * 10**3 + ser * 10 + yd).astype(np.int64)
    docnum = np.char.mod("%014d", docnum_int)          # zero-pad so single-digit FLCs keep 14 chars

    # --- identity columns: sample indexes into the pools ---------------------
    # Sampling with replacement means the same email / device / network shows
    # up on many rows, which is what a real customer table looks like.
    ei = rng.integers(0, len(pools["email"]), n)
    pi = rng.integers(0, len(pools["phone"]), n)
    ni = rng.integers(0, len(pools["ipv4"]), n)        # one index drives BOTH IPv4 and IPv6
    di = rng.integers(0, len(pools["device"]), n)
    ai = rng.integers(0, len(pools["acct"]), n)

    # --- routing number: prepaid card vs bank ------------------------------
    prepaid = rng.random(n) < PREPAID_SHARE
    rtn = np.where(prepaid, rng.choice(pools["prepaid_rtn"], n, p=pools["prepaid_w"]),
                   rng.choice(pools["bank_rtn"], n, p=pools["bank_w"]))

    # --- transaction amounts -----------------------------------------------
    status = rng.choice(STATUS, n, p=STATUS_W)
    trans_amt = np.round(np.exp(rng.normal(7.4, 0.9, n)), 2)     # lognormal: long right tail, median ~$1,640
    trans_amt = np.clip(trans_amt, 25, 85000)

    def disc(p, lo, hi):
        """A discount present in fraction p of rows, worth lo..hi of the gross amount."""
        has = rng.random(n) < p
        return np.round(np.where(has, trans_amt * rng.uniform(lo, hi, n), 0.0), 2)

    d1 = disc(.45, .02, .12)      # 45% of rows, 2-12% off
    d2 = disc(.22, .01, .08)      # 22% of rows, 1-8% off
    d3 = disc(.08, .005, .05)     # 8% of rows, 0.5-5% off
    adj = np.round(np.maximum(trans_amt - d1 - d2 - d3, 0.0), 2)
    refund = np.round(np.where(rng.random(n) < .11, adj * rng.uniform(.05, 1.0, n), 0.0), 2)  # 11% refunded

    return pd.DataFrame({
        "DocNum": docnum,
        "UUID": make_uuid4(rng, n),                   # fresh per row -> unique
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


# ============================================================================ #
# Generate                                                                      #
# ============================================================================ #

def run_generation(target_gb, rows, out_path, seed, chunk, pool_scale, today_str):
    """Build the pools, then write the CSV chunk by chunk until it reaches
    `target_gb` (or `rows` rows).  Returns the number of rows written."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    t0 = time.time()

    # --- routing-number pools + lookup table --------------------------------
    bank_rtn, bank_names, bank_w = make_routing_pool(rng, NAMED_BANKS)
    pre_rtn, pre_names, pre_w = make_routing_pool(rng, PREPAID_BRANDS)
    assert all(aba_is_valid(r) for r in np.r_[bank_rtn, pre_rtn])
    # The lookup CSV maps every routing number back to its institution and
    # type, so you can join it on later (e.g. to add an is_prepaid flag).
    lookup = pd.DataFrame({"RoutingNumber": np.r_[bank_rtn, pre_rtn], "Institution": np.r_[bank_names, pre_names],
                           "Type": ["bank"] * len(bank_rtn) + ["prepaid_card"] * len(pre_rtn),
                           "Weight": np.r_[bank_w * (1 - PREPAID_SHARE), pre_w * PREPAID_SHARE]})
    lookup.to_csv(os.path.splitext(out_path)[0] + "_routing_lookup.csv", index=False)

    # --- identity pools --------------------------------------------------
    # Pool sizes are roughly "number of distinct customers/devices/networks".
    # With ~18.7M rows and ~2M emails, each email appears ~9 times on average.
    s = pool_scale
    ipv4, ipv6 = make_networks(rng, int(900_000 * s))
    pools = dict(bank_rtn=bank_rtn, bank_w=bank_w, prepaid_rtn=pre_rtn, prepaid_w=pre_w,
                 email=make_emails(rng, int(2_200_000 * s)), phone=make_phones(rng, int(2_000_000 * s)),
                 ipv4=ipv4, ipv6=ipv6, device=make_device_ids(rng, int(1_800_000 * s)),
                 acct=make_account_numbers(rng, int(2_000_000 * s)))
    print(f"pools ready in {time.time()-t0:.1f}s", flush=True)

    # --- write loop ------------------------------------------------------
    target = target_gb * 1024**3
    written = 0; n_rows = 0; first = True
    with open(out_path, "w", newline="") as fh:
        while True:
            if rows is not None:                       # fixed row count mode
                n = min(chunk, rows - n_rows)
                if n <= 0: break
            else:                                      # fill-to-size mode
                n = chunk
                if written and n_rows:
                    # bytes per row so far tells us how many more rows fit
                    n = int(min(chunk, max(0, (target - written) / (written / n_rows))))
                    if n <= 0: break
            df = build_chunk(rng, n, pools, today)
            df.to_csv(fh, index=False, header=first, lineterminator="\n")   # header only on the first chunk
            first = False
            written = fh.tell(); n_rows += n
            print(f"rows={n_rows:,}  size={written/1024**3:.2f} GB  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done: {n_rows:,} rows, {written/1024**3:.2f} GB -> {out_path}")
    return n_rows


# ============================================================================ #
# Preview & sanity checks                                                       #
# ============================================================================ #

def sanity_check(out_path, nrows=200_000):
    """Read the first `nrows` rows back and verify the column rules."""
    chk = pd.read_csv(out_path, dtype=str, nrows=nrows)
    ts = pd.to_datetime(chk.SubmissionDate)
    print("Julian day matches date:", (chk.DocNum.str[5:8].astype(int) == ts.dt.dayofyear).all())
    print("Year digit matches date:", (chk.DocNum.str[13].astype(int) == ts.dt.year % 10).all())
    print("UUIDs unique:", chk.UUID.is_unique, "| v4:", all(uuid.UUID(u).version == 4 for u in chk.UUID[:5000]))
    print("ABA check digits valid:", all(aba_is_valid(r) for r in chk.RoutingNumber.unique()))
    print("Phone NANP format:", chk.Phone.str.fullmatch(r"[2-9][0-8]\d-[2-9]\d\d-\d{4}").all())
    print("One IPv6 per IPv4:", chk.groupby("IPv4").IPv6.nunique().max() == 1)
    print(chk.groupby("ReportingYear").SubmissionDate.agg(["min", "max"]))
    print(chk.head(10).to_string())


# ============================================================================ #
# Split into chunks                                                             #
# ============================================================================ #

def split_csv(path, chunk_mb=25, parts_dir="parts", header_each=True, read_mb=8):
    """Stream `path` into line-aligned parts of at most `chunk_mb` MB.

    Rows are never cut in half: the file is read in `read_mb` MB blocks, only
    complete lines are emitted, and a part is closed as soon as the next line
    would push it past the limit.  Parts are named <stem>_part0001.csv, ...

    header_each=True  -> every part starts with the header row (each part is
                         a valid CSV on its own).
    header_each=False -> only part 1 has the header (parts can be `cat`-ed
                         back together without duplicate headers).
    """
    limit = int(chunk_mb * 1024 * 1024)
    os.makedirs(parts_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time(); n = 0; out = None; size = 0

    def open_part():
        """Start a new part file (and write the header if required)."""
        nonlocal n, out, size
        n += 1
        out = open(os.path.join(parts_dir, f"{stem}_part{n:04d}.csv"), "wb")
        hdr = header if (header_each or n == 1) else b""
        out.write(hdr); size = len(hdr)

    with open(path, "rb") as fh:
        header = fh.readline()                       # first line of the source file
        carry = b""                                  # partial last line from the previous block
        while True:
            buf = fh.read(read_mb * 1024 * 1024)
            if not buf:                              # EOF: flush whatever partial line is left
                lines = carry + (b"\n" if carry and not carry.endswith(b"\n") else b"")
                carry = b""
            else:
                buf = carry + buf
                cut = buf.rfind(b"\n") + 1           # position after the last complete line
                lines, carry = buf[:cut], buf[cut:]
            pos = 0
            while pos < len(lines):
                if out is None:
                    open_part()
                # last newline that still fits in the current part
                end = lines.rfind(b"\n", pos, min(len(lines), pos + (limit - size))) + 1
                if end <= pos:                       # next line doesn't fit -> close and start a new part
                    out.close(); out = None; continue
                out.write(lines[pos:end]); size += end - pos; pos = end
            if not buf:
                break
    if out is not None:
        out.close()
    print(f"wrote {n} parts of <= {chunk_mb} MB to {parts_dir}/ in {time.time()-t0:.0f}s")
    return n


# ============================================================================ #
# Entry point                                                                   #
# ============================================================================ #

def main():
    # Command-line flags default to the configuration block at the top, so
    # `python3 gen_dataset.py` with no arguments behaves exactly like the notebook.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-gb", type=float, default=TARGET_GB)
    ap.add_argument("--rows", type=int, default=ROWS, help="exact row count (overrides --target-gb)")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--pool-scale", type=float, default=POOL_SCALE)
    ap.add_argument("--today", default=TODAY)
    ap.add_argument("--no-split", action="store_true", help="skip splitting into parts")
    ap.add_argument("--chunk-mb", type=int, default=CHUNK_MB)
    ap.add_argument("--parts-dir", default=PARTS_DIR)
    ap.add_argument("--no-header-in-parts", action="store_true", help="header only in part 1")
    args = ap.parse_args()

    # 1-2. generate
    run_generation(args.target_gb, args.rows, args.out, args.seed, args.chunk, args.pool_scale, args.today)

    # 3. verify
    sanity_check(args.out)

    # 4. split
    if SPLIT and not args.no_split:
        parts_dir = args.parts_dir or os.path.splitext(args.out)[0] + "_parts"
        split_csv(args.out, args.chunk_mb, parts_dir, header_each=not args.no_header_in_parts)


if __name__ == "__main__":
    main()
