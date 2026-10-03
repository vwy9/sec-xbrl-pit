# sec-xbrl-pit

*[中文版](README.zh.md)*

Two questions, answered on entirely public SEC data:

1. **How often do US financial statement figures get rewritten by later filings?**
   By how much, after how long, and how many arrive through a formal amendment.
2. **Can a program that passes a truncation test still be using data it could
   not have had at the time?**

The second question is why this repository exists. A truncation test checks
that a function `f` is causal with respect to its input tensor `D` — that
`y_t = f(D_0:t)` for all `t`. It says nothing about whether `D` itself is
`F_t`-measurable. **A revision rewrites the past, not the future, so a program
fed restated fundamentals passes truncation unchanged.**

## Results

Window: **2009Q2–2026Q2, 69 quarters of the SEC Financial Statement Data
Sets.**

```
80,960,707 observations   |   41,769,440 business keys
15,055 companies          |   7,664 tags   |   400,076 filings
5.7 GB of source ZIPs  ->  1.0 GB of columnar storage
```

### Restatement census

| | |
|---|---|
| Facts reported by two or more filings | 22,769,445 |
| Facts whose value changed | 2,113,697 (**9.28%**) |
| Changed by more than 1% in relative terms | 6.90% |
| **Share of the 2,326,458 revision events arriving with no formal amendment** | **92.2%** |
| Median revision lag, from first publication | **364 days** |
| Median first-to-final relative magnitude | 9.3% (8.7% excluding unit-scale jumps) |

Three findings worth stating separately:

- **Revisions arrive on the disclosure calendar.** Measured from a figure's
  first publication, **65.7%** of revisions arrive 300–399 days later — the
  comparative column of the following year's report. Annual figures have a
  second peak at **two years** (26.6% of their revisions fall at 700–759 days),
  because annual income and cash-flow statements present up to three years.
  Only 5.9% arrive within 60–129 days, at the next quarterly report. Revisions
  almost never arrive at arbitrary moments; they arrive with a later periodic
  report. That structure is itself the explanation for why monitoring only
  10-K/A filings misses more than nine tenths of them.

  *Which lag you measure matters.* Counted from the previous filing that
  carried the figure, the lag shows a spurious peak near 100 days: a prior
  year-end balance reappears in each of the next three 10-Qs, so a change in
  the next 10-K lands about 100 days after the last 10-Q but nearly a year
  after the figure was first published. The headline numbers therefore count
  from first publication. Both measures are kept (`lag_since_first_days`,
  `lag_days`); `stats_lag_bins.csv` gives the shares.

- **Revision rates depend sharply on which universe you rank over, and the two
  answers differ by more than a factor of two.** Among the *most frequently
  reported* fields, the highest rates are `GeneralAndAdministrativeExpense` and
  `OperatingIncomeLoss` at **15.4%** — roughly one operating-income figure in
  six is later changed. Across *all* fields with at least 1,000 facts reported more than once,
  the highest are discontinued-operations and real-estate items:
  `DiscontinuedOperationIncomeLoss…` at **42.1%**, `RealEstateRevenueNet` at
  33.1%. The latter has a clear mechanism: when a segment is discontinued,
  every prior period is restated to move it below the line.

- **The tail is long.** The 99th percentile is 881 days after first
  publication; the longest single revision arrived **4,295 days** — nearly
  twelve years — after the original.

![Restatement census](figures/fig_census.png)

### Vintage differential demonstration

Three implementations of one cross-sectional signal task (a fourth, `restated`, is run by `scripts/restatement_control.py`; see below), run against two
worlds built from the same observation table:

```
world_pit(t) = observations with filed <= t     D_PIT
world_rev    = all observations                 D_rev
```

| Program | Period alignment | Filing-date alignment | Truncation | Λ_cell | Λ_rank | Verdict |
|---|---|---|---|---|---|---|
| `correct` | `ddate <= t` | `filed <= t` | pass | **0.000** | 0.0000 | clean |
| `restated`† | `ddate <= t` | `filed <= t` (period only; value read from the latest version) | **pass** | 0.952 | 0.0070 | **silent leak** |
| `naive` | `ddate <= t` | none | **pass** | **1.000** | 0.0829 | **silent leak** |
| `lookahead` | `ddate <= t+1y` | none | fail | 1.000 | 0.3194 | double failure |

† `restated` is run by `scripts/restatement_control.py`, not by `secpit.cli demo`.

`naive` makes a mistake practitioners actually make: **it aligns on the
reporting period rather than the filing date.** It never reads a fact with
`ddate > t`, so it passes the truncation test — which is precisely the cell
this repository exists to exhibit.

`lookahead` exists only to show the truncation test is not vacuous. Without a
control that fails it, "both programs passed truncation" cannot be
distinguished from "the truncation test does nothing."

**Two channels, separated.** `naive` leaks through two channels that truncation
cannot see: at a quarter-end it reads that quarter's figures before they are
filed (publication lag), and it reads the latest revision of every figure
(restatement). `scripts/restatement_control.py` separates them. A program
`restated` that picks the period exactly as `correct` does, but reads its value
from the latest version, isolates restatement: it passes truncation with
Λ_rank = 0.0070 (Λ_cell = 0.952), and both fall to exactly 0 when every fact is
frozen at its first-filed value. The same no-revision control leaves `naive` at
Λ_rank = 0.0715 of 0.0829: with as-of dates at quarter-ends, most of its leak is
publication lag. That depends on the as-of dates. At a quarter-end none of that
quarter's reports are filed yet, so the lag channel is at its widest; move every
as-of date 60 days later and `naive` falls to 0.0308, of which the no-revision
control keeps 0.0178, while `restated` rises to 0.0128 — the two channels are
then of similar size. The script prints both settings.

![Vintage differential](figures/fig_lambda.png)

An empirical regularity falls out of this: **filtering on `filed` in practice implies
filtering on `ddate`** — a period can almost never be filed before it has
ended (275 of 3.47 million rows in the three demo fields carry a period end
after their filing date). A program that filters on `filed` therefore passes
truncation in practice, and no program here lands in the cell "truncation
fails, vintage differential passes".

Where each program lands in the 2×2 table is fixed by how it is written, not
discovered by the experiment: truncation cuts on `ddate`, and `naive` never
reads past `ddate <= t`, so it must pass; `correct` filters on `filed <= t`
itself, so it sees the same rows in both worlds and its Λ is zero by
construction. The demonstration is an existence proof that turns the
structural claim into runnable code. Its empirical content is the size of Λ
and the split between the two channels.

## Reproducing

```
export SECPIT_USER_AGENT="Your Name you@example.com"   # SEC asks every request to declare a contact
python3 -m secpit.cli build        # download 69 quarters, parse, assert primary key
python3 -m secpit.cli census       # detect revisions, build census, compute statistics
python3 -m secpit.cli renames      # derive taxonomy equivalence classes, compare
python3 -m secpit.cli crosscheck   # sampled validation against an independent source
python3 -m secpit.cli demo         # two worlds x three programs x two tests
python3 -m secpit.cli figure       # both figures
python3 -m pytest tests/ -q        # 48 unit tests
python3 scripts/restatement_control.py   # lag vs restatement, separated
```

Measured peak resident memory on a 17 GB machine: `build` 3.30 GB,
`census` 3.42 GB, `demo` 1.37 GB, `figure` 0.31 GB.

## Caveats

**Visibility is approximated by `filed`, a date-granularity acceptance date,
and the approximation is optimistic.** Of 400,076 filings, 56.8% were accepted
after 16:00. EDGAR rolls 33,707 of those to the next business day, so their
`filed` value is already correct; but **48.4% were accepted after the close and
still carry the same day's `filed`**. Those were not obtainable intraday yet are
treated as visible that day. The timestamped `accepted` column is retained in
the observation table for anyone wanting a stricter rule.

**Both censoring effects are small, and both push the rate down.** The four
censoring regimes give 9.283% (all) / 9.339% (drop right-censored) / 9.337%
(drop left-censored) / 9.393% (drop both). A fact is left-censored when its
first observation is not its original filing: either it falls in the window's
first quarter (1,315 facts), or the figure first appears in a comparative
column, so the filing that originally reported its period is not in the table
(1,483,697 facts, 6.5% of the denominator). The second case comes from the
2009–2011 XBRL phase-in, IPOs, switches from 20-F to 10-K and late adoption of a
standard tag. Looking only at the first case would make left-censoring look
negligible. Those facts are revised at 8.51%,
below the rest, because a missing original hides the first change; the headline
9.28% is therefore conservative.

**Rates shift with the period covered.** On three 12-quarter windows built the
same way, the rate is 9.06% (2014Q3–2017Q2), 7.42% (2018Q3–2021Q2) and 5.93%
(2023Q3–2026Q2). The windows have the same length and the same censoring, so
the differences come from the period, not from window length: an evaluation
built on recent filings alone sees fewer revisions than the full history
implies. Each window is rebuilt from the same ZIPs in a directory of its own
(`--root` with `data/raw` pointing at the downloaded ZIPs, `--quarters` listing
the twelve quarters), running `build` and then `census`.

**Taxonomy renames were measured, not assumed.** A renamed concept splits one
fact into two unrelated keys, hiding any revision that crosses the rename.
`secpit/renames.py` derives equivalence classes from the data itself — the
decisive criterion is that a true rename shows a high rate of *exact value
agreement* on the same `(cik, ddate, qtrs, uom)`, which cleanly separates real
renames (35–92% agreement on the known cases used for calibration; the 27
accepted pairs range from 41% to 98%) from coincidental successors (~0.1%). Correcting for
it moves the headline from **9.2830% to 9.3103%, a relative increase of 0.29%**.
Merging tags has a trap of its own: a filing that reports both the old and the
new tag for the same period carries two rows under one merged key, and their
difference would be counted as a zero-day revision. Such pairs are resolved by
the same rule as any filing that contradicts itself: identical values are kept
once, conflicting ones (32,973 key–filing groups) are dropped and counted.

**These numbers describe the Financial Statement Data Sets, not the filings
themselves — but a second SEC pipeline now bounds how much that matters.**
SEC's own disclaimer states it cannot guarantee the data sets' accuracy and that
errors may have been introduced during extraction, so FSDS alone cannot separate
a genuine restatement from an extraction artifact. A sampled cross-check against
Company Facts, an independently maintained SEC endpoint, gives:

| | |
|---|---|
| Companies sampled (seed 0, reproducible) | 200 |
| Our observations matched in both sources | 811,097 (93.9% of ours) |
| **Value agreement on matched observations** | **99.999%** (5 conflicts) |
| The 5 conflicts | 4 are four-decimal rounding of stated interest rates; 1 differs by 3.1% |

At 811,097 observations across 200 randomly drawn companies, systematic
extraction error large enough to affect the census is implausible, though the
sample covers only 200 of 15,055 companies (1.33%). The
6.1% of our observations with no counterpart are not disagreements; they are
rows Company Facts does not carry under the same normalised key. Both sources
are compiled by SEC from the same XBRL filings, so agreement rules out errors
introduced in compiling the data sets, not errors the filer made when tagging.

**Λ_cell saturates for this task.** The signal is cross-sectionally
z-scored, so any single revised input contaminates every output cell through
the mean and standard deviation; `naive` scores exactly 1.000. Λ_rank is the
graded measure here (`naive` 0.083, `lookahead` 0.319).

**Six data hazards** are handled explicitly: dimensional breakdown rows,
custom tags, unit-scale jumps, null values, zero first values, and filings that
contradict themselves (26 rows in 81 million report the same fact twice with
different values; these are dropped and counted).

**Other boundaries**: 10-K / 10-Q and their amendments only; consolidated rows
only (about 47% of rows are dimensional breakdowns and are discarded); standard
us-gaap tags only.

**FSDS was chosen over Company Facts as the primary source for
reproducibility.** Quarterly FSDS ZIPs are fixed once published; Company Facts
is a live snapshot rebuilt continuously, so the same code would produce
different results three months from now.

## Layout

```
secpit/
  config.py      constants and paths; the single home for every magic number
  fetch.py       downloads quarterly ZIPs; manifest-driven skipping, atomic rename
  parse.py       header assertions, five filters with count conservation, partitioned write
  store.py       primary-key assertion, as_of / latest / diff_worlds
  revisions.py   revision detection and census, partitioned by cik bucket
  stats.py       twelve statistics tables, most across four censoring regimes
  renames.py     derives taxonomy equivalence classes from the data
  crosscheck.py  sampled validation against Company Facts
  demo.py        data handle (with its deliberate trap), three programs, both tests
  figures.py     census figure, vintage-differential figure
  cli.py         build / census / renames / crosscheck / demo / figure
scripts/
  kraft_heinz_case.py     one known restatement, checked against Company Facts
  restatement_control.py  publication lag and restatement, separated
figures/         both figures (deliverables; tracked)
data/            raw, parquet, out (intermediate; rebuilt by the CLI, not tracked)
tests/           48 unit tests
```

The observation table is partitioned two levels deep as
`filed_year=YYYY / cik_bucket=B`. Each key serves a different access pattern:
`filed_year` lets the primary-key assertion and the acceptance-time statistic
run one year at a time (a filing is accepted in exactly one quarter, so it lands
in exactly one year partition); `cik_bucket` turns the revision-detection
partition filter into genuine partition pruning.

## Environment

miniconda base, Python 3.13. polars 1.40, pandas 3.0, numpy, matplotlib 3.11,
requests, tqdm; pytest for the test suite. No pyarrow, scipy, or DuckDB.

## Data and compliance

Data comes from the
[SEC Financial Statement Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets).
As a work of the United States Government it is not subject to copyright
protection in the United States.

Requests follow SEC's
[EDGAR access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data):
every request carries a declared User-Agent with a contact address, and the
rate stays far below the stated 10 requests/second ceiling. Source ZIPs are not
tracked in version control.

> **SEC disclaimer**: the data sets are derived from information provided by
> individual registrants; the Commission cannot guarantee their accuracy,
> inaccuracies may have been introduced during extraction and compilation, and
> the data sets are not a substitute for the filings themselves.

## License

Code is MIT licensed; see `LICENSE`. The SEC data it processes is public domain.
