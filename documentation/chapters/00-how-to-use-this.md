---
part: Orientation
status: reference
updated: 2026-09-07
---

# How to use this

This is the working record for **bittle-agent** — plans, notes on fixes, what
changed, what is broken, what got verified. It is the thing a developer reads
first, including the developer who is you in three weeks with none of today's
context.

It exists because the alternative kept failing. Work was tracked in chat
transcripts that scroll away, in hosted reports that expire, and in loose
`PLAN-*.md` files nobody could find. A plan that only one person can open is a
plan that gets re-derived from scratch by the next person to touch the code.

## The rule

**At the end of every session, update this documentation.**

If a plan was approved and only half-built, **the plan goes in anyway**, with
`status: in-progress` and an explicit note on what is left. A half-finished plan
that is written down is a handoff. A half-finished plan that is not is a mess
for whoever opens the file next.

Three things follow from that:

1. **Plans land here once they are solid and approved — not before.** A draft
   being argued over does not belong in the record; it would only be edited over
   and over. Approval is the moment it becomes a chapter.
2. **Every update reviews what is already here.** Plans that shipped, ideas that
   stopped being relevant, problems that resolved themselves — mark them.
   A document that only ever grows stops being read.
3. **Nothing is deleted for being stale.** It gets `status: superseded` and one
   line saying what replaced it. A deleted plan cannot tell the next developer
   why it was dropped, and "why not" is the expensive question.

## Writing a chapter

One file in `chapters/`. That is the entire job — there is no manifest to edit,
no renumbering, no HTML:

```markdown
---
part: Plans
status: approved
updated: 2026-09-07
review-by: 2026-09-09
---

# Short title saying what this is

What the problem is, what we decided, and how we will know it worked.
```

The `NN-` prefix is a sort key and is stripped from the title. `part` decides
which section of the page it lands in; a part that does not exist yet is created
for it.

### The fields

| Field | Meaning |
|---|---|
| `part` | Which section of the page. **Required.** |
| `status` | Where it is in its life. See below. |
| `updated` | `YYYY-MM-DD`, when the content was last true. |
| `review-by` | `YYYY-MM-DD`. The page marks it overdue after this date. |
| `owner` | The service or area it is about, if not this repo. |
| `supersedes` | The chapter this one replaces. |

A misspelled field is a hard build error, not a shrug — `stat:` instead of
`status:` would silently drop the chapter out of the staleness review forever.

### The statuses

| Status | Means |
|---|---|
| `approved` | Agreed, not started. |
| `in-progress` | Partly built. **Say what is left.** |
| `blocked` | Started and stuck. Say on what. |
| `shipped` | Built. Say what proved it. |
| `verified` | Shipped *and* confirmed in the running system. |
| `superseded` | Replaced. Name the replacement. |
| `abandoned` | Deliberately not doing it. **Say why** — this is the valuable one. |
| `reference` | Standing documentation, not a plan. Never goes stale. |

`approved`, `in-progress` and `blocked` are claims about the *future*, so they
rot; the daily review looks at exactly those. The rest are claims about the past
and are left alone.

## Building and reading it

```bash
python3 documentation/build_docs.py          # rebuild index.html
python3 documentation/build_docs.py --check  # non-zero if the page is stale
python3 documentation/build_docs.py --status # front matter as JSON
python3 documentation/serve.py --open        # read it locally on :8900
```

Markdown is the source of truth. `index.html` is generated and **must never be
hand-edited** — the next build overwrites it. A `Stop` hook rebuilds the page
automatically when a chapter changes, so a stale page is not something you
normally have to think about.

## What does not belong here

- **A standalone `.md` somewhere else in the repo.** It will not be found.
- **A hosted report or artifact.** It expires from reach and cannot be diffed.
- **Anything in a repo that serves no documentation.** Write it where it can be
  read, and name the other service in the text instead.
