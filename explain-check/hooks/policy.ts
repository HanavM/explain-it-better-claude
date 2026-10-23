// Pure decision logic for explain-check: which turns count as explanations,
// when to ask, and how to read the dialog's answer. No `$` in here, so the
// tests can exercise it directly.

export const DAY_MS = 86_400_000

export type Settings = {
  minChars: number
  minCharsWithoutTools: number
  minGapMs: number
  maxPerDay: number
  baseRate: number
}

export const DEFAULTS: Settings = {
  minChars: 500,
  minCharsWithoutTools: 1500,
  minGapMs: 20 * 60_000,
  maxPerDay: 6,
  baseRate: 0.35,
}

export type Throttle = {
  lastAskAt: number
  day: number
  asksToday: number
  skipStreak: number
  paused: boolean
}

export const FRESH: Throttle = { lastAskAt: 0, day: 0, asksToday: 0, skipStreak: 0, paused: false }

// A turn is worth asking about when it is a substantial message after real
// work, or a long standalone answer (concept explanations land here). Short
// replies that are mostly a question back to the user are not explanations.
export function isExplanation(answer: string, toolCalls: number, s: Settings = DEFAULTS): boolean {
  const text = answer.trim()
  if (text.length < s.minChars) return false
  if (toolCalls === 0 && text.length < s.minCharsWithoutTools) return false
  if (text.length < 800 && text.endsWith('?')) return false
  return true
}

// 0..1: how dense the message is in things this user tends to trip on
// (code identifiers, acronyms, symbols, unexplained numbers).
export function jargonScore(answer: string): number {
  const backticks = (answer.match(/`[^`\n]+`/g) ?? []).length
  const acronyms = (answer.match(/\b[A-Z]{2,6}s?\b/g) ?? []).length
  const symbols = (answer.match(/[α-ωΑ-Ω†‡≤≥≈±∑√]/g) ?? []).length
  const identifiers = (answer.match(/\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z]+\b/g) ?? []).length
  const decimals = (answer.match(/\b\d+\.\d+\b/g) ?? []).length
  const per1k = ((backticks + acronyms + symbols + identifiers + decimals) * 1000) / Math.max(answer.length, 1)
  return Math.min(1, per1k / 15)
}

// A new day clears the count and the skip streak, so a bad day of skipping
// never silences the checks for good.
export function rollDay(t: Throttle, now: number): Throttle {
  const day = Math.floor(now / DAY_MS)
  return day === t.day ? t : { ...t, day, asksToday: 0, skipStreak: 0 }
}

// Each skip in a row doubles the gap (capped at 8x) and halves the chance,
// so dismissing the dialog makes it back off on its own.
export function askProbability(jargon: number, t: Throttle, s: Settings = DEFAULTS): number {
  const p = s.baseRate + (1 - s.baseRate) * 0.8 * jargon
  return p / 2 ** Math.min(t.skipStreak, 3)
}

export function shouldAsk(now: number, t: Throttle, jargon: number, roll: number, s: Settings = DEFAULTS): boolean {
  const cur = rollDay(t, now)
  if (cur.paused) return false
  if (cur.asksToday >= s.maxPerDay) return false
  if (now - cur.lastAskAt < s.minGapMs * 2 ** Math.min(cur.skipStreak, 3)) return false
  return roll < askProbability(jargon, cur, s)
}

// Taken when a check is scheduled, not when it closes, so a second session
// (or the next turn) cannot schedule another one while this one is open.
export function reserve(t: Throttle, now: number): Throttle {
  const cur = rollDay(t, now)
  return { ...cur, lastAskAt: now, asksToday: cur.asksToday + 1 }
}

export function settle(t: Throttle, skipped: boolean): Throttle {
  return { ...t, skipStreak: skipped ? t.skipStreak + 1 : 0 }
}

// The dialog joins picked labels with commas, so labels must not hold any.
export function cleanLabel(raw: string): string {
  const one = raw.replace(/[\n\r]+/g, ' ').replace(/,/g, ' -').replace(/\s+/g, ' ').trim()
  return one.length > 80 ? one.slice(0, 77).trimEnd() + '...' : one
}

export const NONE = 'None of it really'

// Reads the splitter model's reply: a JSON array of strings, possibly wrapped
// in prose or a code fence. Falls back to the message's own first lines.
export function parsePoints(reply: string | undefined, answer: string): string[] {
  let points: string[] = []
  const match = reply?.match(/\[[\s\S]*\]/)
  if (match) {
    try {
      const arr = JSON.parse(match[0])
      if (Array.isArray(arr)) points = arr.filter((x): x is string => typeof x === 'string')
    } catch {}
  }
  if (points.length === 0) points = fallbackPoints(answer)
  const seen = new Set<string>()
  return points
    .map(cleanLabel)
    .filter(p => p.length > 0 && p !== NONE && !seen.has(p) && seen.add(p))
    .slice(0, 3)
}

function fallbackPoints(answer: string): string[] {
  return answer
    .split('\n')
    .map(l => l.replace(/^[#>*\-\d.)\s]+/, '').replace(/\*\*/g, '').trim())
    .filter(l => l.length > 15 && !l.startsWith('|') && !l.startsWith('```'))
    .slice(0, 3)
}

export type Reading = { understood: string[]; notUnderstood: string[]; none: boolean; other: string }

// `answer` is what `$.ui.ask` resolved: picked labels comma-joined, and/or the
// text typed under "Other".
export function readAnswer(answer: string, points: string[]): Reading {
  if (answer.trim() === '') return { understood: [], notUnderstood: points, none: true, other: '' }
  let rest = answer
  const picked = new Set<string>()
  for (const label of [...points, NONE].sort((a, b) => b.length - a.length)) {
    if (rest.includes(label)) {
      picked.add(label)
      rest = rest.replace(label, '')
    }
  }
  const other = rest.replace(/(^[\s,]+)|([\s,]+$)/g, '').replace(/,(\s*,)+/g, ',').trim()
  const none = picked.has(NONE)
  return {
    understood: none ? [] : points.filter(p => picked.has(p)),
    notUnderstood: none ? points : points.filter(p => !picked.has(p)),
    none,
    other,
  }
}

export const SPLIT_SYSTEM = `You split a coding agent's message into the 3 most important things it tells the reader, so the reader can tick which ones they understood.

Rules:
- Pick what matters most to the reader: the result or verdict, what changed, what they need to do or decide, and any caveat that changes what they should do.
- Each item is a short label of at most 12 words.
- Keep the message's own key terms, names, symbols and numbers exactly as written. Do NOT simplify, define or explain them: the point is to test whether the reader understood the message as written.
- No commas in labels.
- Reply with a JSON array of strings only.`
