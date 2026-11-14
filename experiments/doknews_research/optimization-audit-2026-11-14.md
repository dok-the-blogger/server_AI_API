# Hybrid research optimization, 14 November 2026

Scope: three existing hourly profiles (AI, videogames, crypto). Keep the
model, editorial criteria, 30-minute cache TTL and schedules. User accepts
slower execution and temporary Flex capacity unavailability. No paid test
research, automatic Standard fallback, or additional accounting pipeline.

## Implemented

- Server requests `gpt-6.1-sol` on Flex with SDK POST retries disabled.
  Flex gets a 600-second client timeout and 1,800-second observation window.
- A confirmed HTTP 429 `resource_unavailable` before session acceptance is
  stored as a deferred failed job. It does not mark signals researched or
  block later scheduled work. Replay preserves that job; unknown outcomes,
  generic rate limits and accepted sessions remain blocked against duplicates.
- Research instructions include the compact research contract and unchanged
  policy, calibration and AI criteria. Editor orchestration, publication and
  memory-management instructions are excluded from the API package.
- Check event date and substantive novelty before deep investigation. Return
  the dossier inline without duplicate files, export bundles or empty tables.
  Reuse unchanged source versions and report precise gaps instead of trying
  to reconstruct unavailable archives or repeatedly retry inaccessible pages.
- Integration instructions preserve original source wording across runs and
  check previous conclusions before forwarding the same event again.

An offline comparison with the installed editorial package measured 71,777
versus 47,112 instruction characters (126,789 versus 83,843 UTF-8 bytes):
34.4% fewer characters. This excludes the variable snapshot and is not a
measured reduction in total billed tokens or daily cost.

## Observed costly investigations

Read stored final dossiers and SQL receipts, not complete step traces.
Therefore repeated reads/tool-call counts cannot be measured from this audit.
Input counts below accumulate session work; they are not unique context size.

| Profile / job | UTC start | Input tokens | Dossier characters | Observations |
| --- | --- | ---: | ---: | --- |
| crypto / b8c6b7dc-cd06-4b18-a15f-08272ab51f70 | Nov 14 07:51 | 4,285,390 | 18,958 | Two held signals; exported dossier, JSON datasets and GraphQL. Reconstructed liquidation history despite missing primary receipts; preexisting depeg required a freshness decision. |
| games / 711c5ff9-01fa-4725-b632-f731994ad32b | Nov 13 20:43 | 2,018,413 | 20,761 | Two finalists and a held statement; duplicated final text in a file and repeated archive/memory coverage caveats. |
| games / d0612fc5-a021-4621-be0c-b9feb050aaf8 | Nov 14 01:42 | 2,004,130 | 15,378 | Single Digital Fairness Act signal held because the event was outside the window without a confirmed new development; substantial contextual analysis nonetheless. |

Some primary verification is necessary. These observations justify focused
instructions, not an assertion that every lengthy step was wasteful or that
we know the savings attributable to individual steps.

The same Fortune robotics URL was sent in regular AI jobs
fb7b50a9-ff96-4505-ac5f-fc709cb97565 (Nov 14 03:16 UTC) and
b5bcd7fe-5f67-4853-b6db-35755bc5c57e (09:13 UTC). Signal ID, version, title
and publication time matched; two differently worded excerpts changed the
content fingerprint. The excerpts describe the same team reassignment.
Do not weaken ledger content comparison: a real edit can share URL/version.
Instead preserve source wording and require a stated factual increment in
the adapter workflow, using previous memory/dossiers to avoid re-investigation.

## Six-hour overview and hourly updates: recommendation, not deployed

The current server already selects only new signals (at most three), and the
editor checks unfinished work and saved finalists before new research. It
does not intentionally commission a full topic survey every hour. A six-hour
cadence alone therefore does not divide costs by six.

Prefer a compact evolving topic brief: established facts, source versions,
open questions, previous rejects/refutations and pending candidates. Refresh
the broad overview roughly every six hours if it improves coverage. Hourly
passes should find new facts, reopen a held candidate only when its condition
changes, and commission targeted verification when needed. Breaking events
must not wait six hours; no new facts means no new paid investigation.

Do not accumulate all work in one ever-growing API conversation: its repeated
history can erase savings. Reuse the compact brief and original evidence
on demand. Existing memory/backlog can support this without another scheduler
or parallel state store. Evaluate Flex and the smaller instructions first.

Per six-hour block, savings require C_overview + 5*C_update < 6*C_hourly,
including unsuccessful investigations and editor follow-up. Compare freshness,
missed significant events and publication quality as well as cost. No measured
price or savings for this proposed cadence are claimed.
