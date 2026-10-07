#!/usr/bin/env python3
"""
Synthetic transaction dataset generator  (standard library only)
================================================================

Produces a large CSV (default ~4.5 GB, ~18.7M rows) of fake tax-style
transactions, then optionally splits it into 25 MB parts.  This script is the
plain-Python twin of ``gen_dataset.ipynb`` -- same sections, same defaults,
same output for the same seed.

**No third-party packages.**  Only the standard library is used:
``random`` for all sampling, ``datetime`` for the filing windows, ``uuid`` for
UUIDs, ``csv`` for reading/writing, and ``multiprocessing`` so the row
generation can use every CPU core.

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
2. Generate rows in chunks of CHUNK rows.  Each chunk gets its own
   deterministic seed (SEED + chunk number), which lets a pool of worker
   processes build chunks in parallel while the file stays reproducible.
   Chunks are written to the CSV in order; stop at TARGET_GB (or ROWS rows).
3. Read the first 200k rows back and verify every column rule.
4. Optionally split the big CSV into line-aligned parts of <= CHUNK_MB each.

Speed: pure Python builds roughly 60-80k rows per second per core, so the full
4.5 GB file takes a few minutes on a multi-core machine (WORKERS = all cores
by default) or ~5 minutes per core if you set WORKERS = 1.
"""

# ============================================================================ #
# Configuration                                                                 #
# ============================================================================ #
# Edit these, or override them on the command line (see --help).

TARGET_GB  = 4.5                 # approximate output size in GB (ignored if ROWS is set)
ROWS       = None                # e.g. 200_000 for a quick test; None = fill to TARGET_GB
OUT_PATH   = "transactions.csv"  # output CSV; a *_routing_lookup.csv is written next to it
SEED       = 42                  # same seed -> identical file every run
CHUNK      = 200_000             # rows per chunk (each chunk is one unit of parallel work)
POOL_SCALE = 1.0                 # multiplier on the size of the identity pools (see below)
TODAY      = "2026-10-06"        # submission dates are capped here (no future-dated rows)
WORKERS    = None                # worker processes; None = number of CPU cores, 1 = no multiprocessing

# Splitting (step 4).  Set SPLIT = False to skip it.
SPLIT               = True
CHUNK_MB            = 25         # max size of each part, in MB
PARTS_DIR           = None       # None -> "<OUT_PATH stem>_parts/" next to the CSV
HEADER_IN_EACH_PART = True       # True: every part is a standalone CSV with a header row

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta


# ============================================================================ #
# Reference pools                                                               #
# ============================================================================ #
# These are the "vocabularies" the generator draws from.  Each has a matching
# weight list so common values appear more often than rare ones.
# (random.choices takes relative weights directly, so nothing needs to sum to 1.)

# --- DocNum building blocks --------------------------------------------------
# File Location Codes: the two leading digits of an IRS DLN identify the
# campus / service center that processed the document.  These are real codes.
FLC_CODES = [7, 9, 17, 18, 28, 29, 37, 38, 49, 56, 66, 70, 76, 77, 80, 90, 94, 95]
FLC_W     = [8, 6, 5, 4, 7, 6, 8, 6, 7, 5, 2, 6, 3, 5, 4, 5, 6, 4]

# Tax class digit.  2 = individual income tax, which dominates; others are rare.
TAX_CLASS   = [2, 1, 3, 4, 5]
TAX_CLASS_W = [80, 6, 6, 4, 4]

# Two-digit document codes (which form / transaction type was filed).
DOC_CODES = [10, 11, 12, 21, 22, 26, 27, 70, 72, 73]
DOC_W     = [22, 22, 18, 10, 8, 6, 6, 4, 2, 2]

# --- Email building blocks ---------------------------------------------------
EMAIL_DOMAINS  = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com",
                  "comcast.net", "att.net", "protonmail.com", "live.com", "msn.com", "verizon.net",
                  "sbcglobal.net", "me.com", "ymail.com", "mail.com"]
EMAIL_DOMAIN_W = [42, 14, 9, 8, 6, 4, 3, 2, 2, 2, 2, 1.5, 1.5, 1, 1, 1]

# Common US first and last names; emails are assembled from these in a few
# styles (first.last, firstlast123, f_last88, ...).
FIRST = ("james john robert michael william david richard joseph thomas charles chris daniel matthew anthony "
         "mark donald steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey "
         "ryan jacob gary nick eric jon stephen larry justin scott brandon ben sam greg frank alex raymond "
         "patrick jack dennis jerry tyler aaron jose adam nathan henry doug zach peter kyle ethan walter noah "
         "mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy betty sandra "
         "margaret ashley kim emily donna michelle carol amanda melissa deborah stephanie rebecca sharon laura "
         "cynthia dorothy amy kathleen angela shirley brenda emma anna pamela nicole samantha katherine "
         "christine helen debra rachel carolyn janet maria catherine heather diane olivia julie joyce victoria "
         "ruth virginia lauren kelly christina joan evelyn judith andrea hannah megan cheryl jacqueline martha "
         "madison teresa gloria sara janice ann kathryn abigail sophia frances jean alice judy isabella julia "
         "grace amber denise danielle marilyn beverly charlotte natalie theresa diana brittany doris kayla").split()
LAST = ("smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez "
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
        "guzman burns crawford olson simpson porter hunter gordon mendez silva shaw snyder mason dixon").split()

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
STATUS   = ["single", "married_fs", "hoh"]
STATUS_W = [52, 33, 15]

# --- IPv4 ranges to avoid ----------------------------------------------------
# First octets that are private, loopback, link-local, documentation, or
# multicast/reserved.  We want every IPv4 to look like a public address.
PRIVATE_FIRST_OCTETS = {0, 10, 127, 100, 169, 172, 192, 198, 203} | set(range(224, 256))
PUBLIC_FIRST_OCTETS = [o for o in range(1, 224) if o not in PRIVATE_FIRST_OCTETS]

# IPv6 global prefixes typical of US ISPs, with weights.
IPV6_PREFIXES  = ["2001", "2600", "2601", "2603", "2605", "2607", "260a", "2620", "2a02", "2a03", "2001:db8"]
IPV6_PREFIX_W  = [14, 18, 18, 10, 8, 10, 4, 6, 4, 4, 4]

# Hour-of-day weights (index = hour 0..23): quiet overnight, busy 9am-9pm.
HOUR_W = [1, 1, .5, .5, .5, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 7, 6, 4, 2]

# CSV column order.
COLUMNS = ["DocNum", "UUID", "ReportingYear", "SubmissionDate", "RoutingNumber", "AccountNumber",
           "Email", "Phone", "IPv4", "IPv6", "DeviceID", "trans_amt", "adj_trans_amt", "refund_amt",
           "customer_status", "discount_1", "discount_2", "discount_3"]


# ============================================================================ #
# Field generators                                                              #
# ============================================================================ #
# Each generator takes a `random.Random` instance `rng` (so everything is
# reproducible from the seed) and returns ONE value, or a list of n values.

def aba_check_digit(d8):
    """Compute the 9th (check) digit of a routing number from its first eight
    digits (a list of 8 ints).  The ABA rule: weight the digits
    3,7,1,3,7,1,3,7,(1) and the total must be a multiple of 10."""
    s = 3 * (d8[0] + d8[3] + d8[6]) + 7 * (d8[1] + d8[4] + d8[7]) + (d8[2] + d8[5])
    return (10 - s % 10) % 10


def aba_is_valid(rtn):
    """True if the 9-digit routing-number string passes the ABA checksum."""
    d = [int(c) for c in rtn]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


def make_routing_pool(rng, banks):
    """Turn a list of (name, prefix, weight) into parallel lists of
    routing numbers, institution names, and sampling weights.

    A bank with weight >= 3 gets three distinct routing numbers (large banks
    really do have many); its weight is shared equally among them.
    """
    names, weights, rtns = [], [], []
    for name, prefix, w in banks:
        if prefix is None:                                   # small institution: random district
            prefix = rng.choice(list(range(1, 13)) + list(range(21, 33)))
        n_rtn = 3 if w >= 3 else 1
        for _ in range(n_rtn):
            while True:                                      # loop until we get an unused number
                d8 = [prefix // 10, prefix % 10] + [rng.randrange(10) for _ in range(6)]
                rtn = "".join(map(str, d8)) + str(aba_check_digit(d8))
                if rtn not in rtns:
                    break
            names.append(name); weights.append(w / n_rtn); rtns.append(rtn)
    return rtns, names, weights


def make_uuid4(rng, upper=False):
    """One RFC-4122 version-4 UUID string, e.g. 8ca5fe05-d593-42f4-88c3-2474c49be7eb.

    Built from 128 random bits taken from OUR seeded rng (uuid.uuid4() would
    use os.urandom and break reproducibility).  uuid.UUID(..., version=4)
    forces the version nibble to 4 and the variant bits to 10xx.
    """
    u = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    return u.upper() if upper else u


def make_phone(rng):
    """One NANP-valid phone number as ###-###-####.

    NANP rules enforced:
      * area code (NPA):  first digit 2-9, second digit 0-8, not N11 (e.g. 911)
      * exchange (NXX):   first digit 2-9, not N11, and we also avoid 555
      * line number:      any 4 digits
    """
    a, b, c = rng.randint(2, 9), rng.randint(0, 8), rng.randrange(10)
    if b == 1 and c == 1: c = 2                              # no N11 area codes
    d, e, f = rng.randint(2, 9), rng.randrange(10), rng.randrange(10)
    if e == 1 and f == 1: f = 2                              # no N11 exchanges
    if d == 5 and e == 5 and f == 5: d = 6                   # no 555 exchange
    return f"{a}{b}{c}-{d}{e}{f}-{rng.randrange(10000):04d}"


def make_email(rng):
    """One email address assembled from the name lists in one of six styles."""
    f, l = rng.choice(FIRST), rng.choice(LAST)
    style = rng.randrange(6)
    if style == 0:   local = f"{f}.{l}"
    elif style == 1: local = f"{f}{l}{rng.randint(1, 1999)}"
    elif style == 2: local = f"{f[0]}{l}{rng.randint(1, 1999)}"       # first initial + last + number
    elif style == 3: local = f"{f}_{l}{rng.randint(55, 99)}"           # birth-year style
    elif style == 4: local = f"{l}.{f}{rng.randint(1, 1999)}"
    else:            local = f"{f}{rng.randint(55, 99)}"
    return local + "@" + rng.choices(EMAIL_DOMAINS, EMAIL_DOMAIN_W)[0]


def make_network(rng):
    """One coherent (ipv4, ipv6) pair.

    The IPv6 is derived from the IPv4's octets (a common ISP prefix, then the
    v4 /16 and /24 encoded as hextets, then a random interface id), so the two
    look like they come from the same provider.
    """
    o1 = rng.choice(PUBLIC_FIRST_OCTETS)                     # never private/reserved
    o2, o3, o4 = rng.randrange(256), rng.randrange(256), rng.randint(1, 254)
    ipv4 = f"{o1}.{o2}.{o3}.{o4}"
    pfx = rng.choices(IPV6_PREFIXES, IPV6_PREFIX_W)[0]
    ipv6 = (f"{pfx}:{(o1 << 8) | o2:x}:{(o3 << 4) | (o4 & 0xF):x}::"
            f"{rng.randrange(65536):x}:{rng.randrange(65536):x}:{rng.randrange(65536):x}:{rng.randrange(65536):x}")
    return ipv4, ipv6


def make_device_id(rng):
    """One device ID in a platform mix:
         34% Windows  -> GUID in braces, uppercase   {46CCB5A7-8CE7-42BF-AD9C-CE05CA1E80B2}
         30% iOS      -> UUID uppercase              D71B2918-7908-4121-9C55-60152EB6FC21
         11% Mac      -> UUID lowercase              b3b22c9f-8cad-45de-970b-ae821f6ead8b
         25% Android  -> 16 hex chars (ANDROID_ID)   b71cbb3c94d37093
    """
    kind = rng.choices((0, 1, 2, 3), (34, 30, 11, 25))[0]
    if kind == 0: return "{" + make_uuid4(rng, upper=True) + "}"
    if kind == 1: return make_uuid4(rng, upper=True)
    if kind == 2: return make_uuid4(rng)
    return f"{rng.getrandbits(64):016x}"


def make_account_number(rng):
    """One account number of 8-12 digits (10 digits most common).
    Leading zeros are allowed, as in real account numbers."""
    length = rng.choices((8, 9, 10, 11, 12), (14, 22, 34, 18, 12))[0]
    return f"{rng.randrange(10 ** length):0{length}d}"


def build_pools(seed, pool_scale):
    """Build the routing-number pools and the identity pools once.

    Pool sizes are roughly "number of distinct customers/devices/networks".
    With ~18.7M rows and ~2.2M emails, each email appears ~9 times on average.
    Every worker process rebuilds the pools from the same seed, so they all
    sample from identical lists.
    """
    rng = random.Random(seed)
    bank_rtn, bank_names, bank_w = make_routing_pool(rng, NAMED_BANKS)
    pre_rtn, pre_names, pre_w = make_routing_pool(rng, PREPAID_BRANDS)
    assert all(aba_is_valid(r) for r in bank_rtn + pre_rtn)

    s = pool_scale
    nets = {}                                                # dict: dedupe on IPv4 -> one IPv6 per IPv4
    for _ in range(int(900_000 * s)):
        v4, v6 = make_network(rng)
        nets.setdefault(v4, v6)
    nets = list(nets.items())

    return dict(
        bank_rtn=bank_rtn, bank_names=bank_names, bank_w=bank_w,
        prepaid_rtn=pre_rtn, prepaid_names=pre_names, prepaid_w=pre_w,
        email=[make_email(rng) for _ in range(int(2_200_000 * s))],
        phone=[make_phone(rng) for _ in range(int(2_000_000 * s))],
        net=nets,
        device=[make_device_id(rng) for _ in range(int(1_800_000 * s))],
        acct=[make_account_number(rng) for _ in range(int(2_000_000 * s))],
    )


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
    """(start, end) datetimes of the filing window for a reporting year,
    with the end capped at `today` so nothing is dated in the future."""
    y = reporting_year + 1
    start = datetime(y, 1, 27)
    end = datetime(y, 10, 15, 23, 59, 59)
    return start, min(end, today)


class DateSampler:
    """Precomputes the window geometry for one reporting year so sampling a
    single timestamp is cheap."""

    def __init__(self, reporting_year, today):
        self.start, end = window_bounds(reporting_year, today)
        self.total_days = (end - self.start).days + 1
        self.apr15 = (datetime(reporting_year + 1, 4, 15) - self.start).days   # day offset of April 15
        self.oct15 = (datetime(reporting_year + 1, 10, 15) - self.start).days  # day offset of October 15

    def sample(self, rng):
        comp = rng.choices((0, 1, 2), (72, 18, 10))[0]                      # which mixture component
        if comp == 0:   d = rng.gauss(self.apr15 - 14, 22)                   # main season
        elif comp == 1: d = rng.uniform(0, self.total_days)                  # background
        else:           d = self.oct15 - rng.expovariate(1 / 9)              # extension rush
        d = min(max(d, 0), self.total_days - 1e-6)                           # keep inside the window
        hour = rng.choices(range(24), HOUR_W)[0]
        sec = hour * 3600 + rng.randrange(3600)                              # random minute/second within the hour
        return self.start + timedelta(days=int(d), seconds=sec)


# ============================================================================ #
# Row builder                                                                   #
# ============================================================================ #

_POOLS = None            # per-process cache of the pools (filled by _init_worker)
_DATES = None            # per-process cache of {reporting_year: DateSampler}


def _init_worker(seed, pool_scale, today_str):
    """Runs once in each worker process: build the pools and date samplers."""
    global _POOLS, _DATES
    _POOLS = build_pools(seed, pool_scale)
    today = datetime.strptime(today_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    _DATES = {y: DateSampler(y, today) for y in (2024, 2025)}


def build_chunk(args):
    """Build one chunk of rows and return it as a single CSV string.

    `args` is (chunk_index, n_rows, seed).  The chunk gets its own Random
    seeded with seed + chunk_index, which is what makes the output identical
    whether chunks are built by one process or many.
    """
    chunk_idx, n, seed = args
    rng = random.Random(seed * 1_000_003 + chunk_idx)
    P, D = _POOLS, _DATES

    # Local aliases: attribute lookups inside a 200k-iteration loop add up.
    choice, choices, random_, randrange = rng.choice, rng.choices, rng.random, rng.randrange
    emails, phones, nets, devices, accts = P["email"], P["phone"], P["net"], P["device"], P["acct"]
    bank_rtn, bank_w, pre_rtn, pre_w = P["bank_rtn"], P["bank_w"], P["prepaid_rtn"], P["prepaid_w"]

    lines = []
    for _ in range(n):
        # --- reporting year and submission timestamp ---------------------------
        ry = 2024 if random_() < .47 else 2025
        ts = D[ry].sample(rng)

        # --- DocNum ------------------------------------------------------------
        # FLC(2) tax class(1) doc code(2) Julian day(3) block(3) serial(2) year digit(1)
        # Julian day and year digit come from the timestamp so they always agree.
        docnum = (f"{choices(FLC_CODES, FLC_W)[0]:02d}"
                  f"{choices(TAX_CLASS, TAX_CLASS_W)[0]}"
                  f"{choices(DOC_CODES, DOC_W)[0]:02d}"
                  f"{ts.timetuple().tm_yday:03d}"           # Julian day 001-366
                  f"{randrange(1000):03d}"                   # block 000-999
                  f"{randrange(100):02d}"                    # serial 00-99
                  f"{ts.year % 10}")                         # last digit of the processing year

        # --- identity columns: sample from the pools ---------------------------
        # Sampling with replacement means the same email / device / network shows
        # up on many rows, which is what a real customer table looks like.
        ipv4, ipv6 = choice(nets)                            # one pick drives BOTH IPv4 and IPv6

        # --- routing number: prepaid card vs bank ------------------------------
        if random_() < PREPAID_SHARE:
            rtn = choices(pre_rtn, pre_w)[0]
        else:
            rtn = choices(bank_rtn, bank_w)[0]

        # --- transaction amounts -----------------------------------------------
        trans = rng.lognormvariate(7.4, 0.9)                 # long right tail, median ~$1,640
        trans = min(max(trans, 25.0), 85000.0)
        d1 = trans * rng.uniform(.02, .12) if random_() < .45 else 0.0    # 45% of rows, 2-12% off
        d2 = trans * rng.uniform(.01, .08) if random_() < .22 else 0.0    # 22% of rows, 1-8% off
        d3 = trans * rng.uniform(.005, .05) if random_() < .08 else 0.0   # 8% of rows, 0.5-5% off
        adj = max(trans - d1 - d2 - d3, 0.0)
        refund = adj * rng.uniform(.05, 1.0) if random_() < .11 else 0.0  # 11% refunded

        # --- assemble the CSV line ---------------------------------------------
        # No value can contain a comma, quote or newline, so a plain join is
        # safe and much faster than csv.writer.
        lines.append(
            f"{docnum},{make_uuid4(rng)},{ry},{ts:%Y-%m-%d %H:%M:%S},{rtn},{choice(accts)},"
            f"{choice(emails)},{choice(phones)},{ipv4},{ipv6},{choice(devices)},"
            f"{trans:.2f},{adj:.2f},{refund:.2f},{choices(STATUS, STATUS_W)[0]},{d1:.2f},{d2:.2f},{d3:.2f}"
        )
    return "\n".join(lines) + "\n"


# ============================================================================ #
# Generate                                                                      #
# ============================================================================ #

def write_routing_lookup(out_path, pools):
    """Write <stem>_routing_lookup.csv mapping every routing number back to its
    institution and type, so you can join it on later (e.g. an is_prepaid flag)."""
    bw = sum(pools["bank_w"]); pw = sum(pools["prepaid_w"])
    with open(os.path.splitext(out_path)[0] + "_routing_lookup.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["RoutingNumber", "Institution", "Type", "Weight"])
        for r, nm, wt in zip(pools["bank_rtn"], pools["bank_names"], pools["bank_w"]):
            w.writerow([r, nm, "bank", f"{wt / bw * (1 - PREPAID_SHARE):.6g}"])
        for r, nm, wt in zip(pools["prepaid_rtn"], pools["prepaid_names"], pools["prepaid_w"]):
            w.writerow([r, nm, "prepaid_card", f"{wt / pw * PREPAID_SHARE:.6g}"])


def run_generation(target_gb, rows, out_path, seed, chunk, pool_scale, today_str, workers=None):
    """Build the pools, then write the CSV chunk by chunk until it reaches
    `target_gb` (or `rows` rows).  Returns the number of rows written.

    Chunks are generated by a pool of worker processes and written in order.
    In fill-to-size mode we don't know the row count up front, so chunks are
    requested in waves of `workers` and the loop stops once the file is big
    enough (the final chunk is sized to land on the target).
    """
    t0 = time.time()
    workers = workers or os.cpu_count() or 1

    # Pools are built in the main process too (for the lookup file + checks).
    _init_worker(seed, pool_scale, today_str)
    write_routing_lookup(out_path, _POOLS)
    print(f"pools ready in {time.time() - t0:.1f}s  (workers={workers})", flush=True)

    target = target_gb * 1024 ** 3
    written = 0; n_rows = 0; chunk_idx = 0

    def next_sizes():
        """Row counts for the next wave of chunks, or [] when we're done."""
        nonlocal chunk_idx
        sizes = []
        for _ in range(workers):
            if rows is not None:                                       # fixed row count mode
                n = min(chunk, rows - n_rows - sum(sizes))
            else:                                                      # fill-to-size mode
                n = chunk
                if written and n_rows:
                    bytes_per_row = written / n_rows
                    remaining = target - written - sum(sizes) * bytes_per_row
                    n = int(min(chunk, max(0, remaining / bytes_per_row)))
            if n <= 0:
                break
            sizes.append(n)
        return sizes

    with open(out_path, "w", newline="") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        if workers == 1:
            # Simple path: no multiprocessing at all.
            while True:
                sizes = next_sizes()
                if not sizes: break
                for n in sizes:
                    fh.write(build_chunk((chunk_idx, n, seed)))
                    chunk_idx += 1; n_rows += n; written = fh.tell()
                    print(f"rows={n_rows:,}  size={written / 1024 ** 3:.2f} GB  elapsed={time.time() - t0:.0f}s", flush=True)
        else:
            with mp.Pool(workers, initializer=_init_worker, initargs=(seed, pool_scale, today_str)) as pool:
                while True:
                    sizes = next_sizes()
                    if not sizes: break
                    jobs = [(chunk_idx + i, n, seed) for i, n in enumerate(sizes)]
                    for n, text in zip(sizes, pool.imap(build_chunk, jobs)):   # imap keeps the order
                        fh.write(text)
                        chunk_idx += 1; n_rows += n; written = fh.tell()
                        print(f"rows={n_rows:,}  size={written / 1024 ** 3:.2f} GB  elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"done: {n_rows:,} rows, {written / 1024 ** 3:.2f} GB -> {out_path}")
    return n_rows


# ============================================================================ #
# Preview & sanity checks                                                       #
# ============================================================================ #

def sanity_check(out_path, nrows=200_000):
    """Read the first `nrows` rows back with the csv module and verify the
    column rules.  Prints True/False for each check plus a preview."""
    with open(out_path, newline="") as fh:
        rows = [r for _, r in zip(range(nrows), csv.DictReader(fh))]

    ts = [datetime.strptime(r["SubmissionDate"], "%Y-%m-%d %H:%M:%S") for r in rows]
    print("DocNum is 14 digits:", all(len(r["DocNum"]) == 14 and r["DocNum"].isdigit() for r in rows))
    print("Julian day matches date:", all(int(r["DocNum"][5:8]) == t.timetuple().tm_yday for r, t in zip(rows, ts)))
    print("Year digit matches date:", all(int(r["DocNum"][13]) == t.year % 10 for r, t in zip(rows, ts)))
    uuids = [r["UUID"] for r in rows]
    print("UUIDs unique:", len(set(uuids)) == len(uuids), "| v4:", all(uuid.UUID(u).version == 4 for u in uuids[:5000]))
    print("ABA check digits valid:", all(aba_is_valid(r) for r in {r["RoutingNumber"] for r in rows}))
    print("Account length 8-12:", all(8 <= len(r["AccountNumber"]) <= 12 for r in rows))
    phone_re = re.compile(r"[2-9][0-8]\d-[2-9]\d\d-\d{4}")
    print("Phone NANP format:", all(phone_re.fullmatch(r["Phone"]) for r in rows))
    v4_to_v6 = {}
    coherent = all(v4_to_v6.setdefault(r["IPv4"], r["IPv6"]) == r["IPv6"] for r in rows)
    print("One IPv6 per IPv4:", coherent)
    for y in ("2024", "2025"):                       # date window per reporting year
        ds = [r["SubmissionDate"] for r in rows if r["ReportingYear"] == y]
        print(f"ReportingYear {y}: {min(ds)} -> {max(ds)}")
    print("\nFirst rows:")
    for r in rows[:5]:
        print("  " + ",".join(r.values()))


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
    print(f"wrote {n} parts of <= {chunk_mb} MB to {parts_dir}/ in {time.time() - t0:.0f}s")
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
    ap.add_argument("--workers", type=int, default=WORKERS, help="worker processes (default: all cores)")
    ap.add_argument("--no-split", action="store_true", help="skip splitting into parts")
    ap.add_argument("--chunk-mb", type=int, default=CHUNK_MB)
    ap.add_argument("--parts-dir", default=PARTS_DIR)
    ap.add_argument("--no-header-in-parts", action="store_true", help="header only in part 1")
    args = ap.parse_args()

    # 1-2. generate
    run_generation(args.target_gb, args.rows, args.out, args.seed, args.chunk, args.pool_scale,
                   args.today, args.workers)

    # 3. verify
    sanity_check(args.out)

    # 4. split
    if SPLIT and not args.no_split:
        parts_dir = args.parts_dir or os.path.splitext(args.out)[0] + "_parts"
        split_csv(args.out, args.chunk_mb, parts_dir, header_each=not args.no_header_in_parts)


if __name__ == "__main__":
    main()
