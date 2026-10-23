import { describe, expect, test } from 'claude-code/testing'

import {
  DAY_MS,
  DEFAULTS,
  FRESH,
  NONE,
  reserve,
  rollDay,
  settle,
  isExplanation,
  jargonScore,
  parsePoints,
  readAnswer,
  shouldAsk,
} from '../hooks/policy'

const long = (n: number) => 'word '.repeat(Math.ceil(n / 5)).slice(0, n)

describe('isExplanation', () => {
  test('short replies are not explanations', () => {
    expect(isExplanation(long(200), 5)).toBe(false)
  })
  test('a substantial message after tool work is', () => {
    expect(isExplanation(long(700), 3)).toBe(true)
  })
  test('a no-tool answer needs to be long', () => {
    expect(isExplanation(long(900), 0)).toBe(false)
    expect(isExplanation(long(1600), 0)).toBe(true)
  })
  test('a short message ending in a question is a question', () => {
    expect(isExplanation(long(600) + '?', 2)).toBe(false)
  })
})

describe('jargonScore', () => {
  test('plain prose scores low, identifier-heavy text scores high', () => {
    expect(jargonScore('The change makes the page load faster for you.') < 0.2).toBe(true)
    expect(jargonScore('`runV2` calls `changedFiles()`; rho_cut=0.80, MAD and BH, ρ ≤ 0.8') > 0.8).toBe(true)
  })
})

describe('shouldAsk', () => {
  const now = 100 * DAY_MS + 5000
  test('respects pause, daily cap and gap', () => {
    expect(shouldAsk(now, { ...FRESH, paused: true }, 1, 0)).toBe(false)
    expect(shouldAsk(now, { ...FRESH, day: 100, asksToday: DEFAULTS.maxPerDay }, 1, 0)).toBe(false)
    expect(shouldAsk(now, { ...FRESH, lastAskAt: now - 60_000 }, 1, 0)).toBe(false)
    expect(shouldAsk(now, FRESH, 1, 0)).toBe(true)
  })
  test('the daily cap resets on a new day', () => {
    expect(shouldAsk(now, { ...FRESH, day: 99, asksToday: DEFAULTS.maxPerDay }, 1, 0)).toBe(true)
  })
  test('skips widen the gap', () => {
    const t = { ...FRESH, day: 100, lastAskAt: now - 25 * 60_000, skipStreak: 2 }
    expect(shouldAsk(now, t, 1, 0)).toBe(false)
    expect(shouldAsk(now, { ...t, skipStreak: 0 }, 1, 0)).toBe(true)
  })
  test('the roll decides within the probability', () => {
    expect(shouldAsk(now, FRESH, 0, 0.3)).toBe(true)
    expect(shouldAsk(now, FRESH, 0, 0.5)).toBe(false)
  })
  test('reserve counts the ask up front and blocks the next one', () => {
    const t1 = reserve(FRESH, now)
    expect(t1.asksToday).toBe(1)
    expect(shouldAsk(now + 1000, t1, 1, 0)).toBe(false)
  })
  test('settle tracks skips, and a new day clears them', () => {
    const t1 = settle(settle(reserve(FRESH, now), true), true)
    expect(t1.skipStreak).toBe(2)
    expect(settle(t1, false).skipStreak).toBe(0)
    expect(rollDay(t1, now + DAY_MS).skipStreak).toBe(0)
  })
})

describe('parsePoints', () => {
  test('reads a fenced JSON array and strips commas', () => {
    const pts = parsePoints('```json\n["Fixed the merge bug, finally", "ρ ≤ 0.80 test drops 12 tasks"]\n```', 'x')
    expect(pts).toEqual(['Fixed the merge bug - finally', 'ρ ≤ 0.80 test drops 12 tasks'])
  })
  test('keeps at most three', () => {
    expect(parsePoints('["a1","b2","c3","d4"]', 'x').length).toBe(3)
  })
  test('falls back to the message lines', () => {
    const pts = parsePoints(undefined, '## Summary\n- **Moved the cache** into the worker process\n| a | b |\n')
    expect(pts).toEqual(['Summary', 'Moved the cache into the worker process'].filter(p => p.length > 15))
  })
})

describe('readAnswer', () => {
  const pts = ['Fixed the merge bug', 'ρ ≤ 0.80 test drops 12 tasks', 'Run make eval next']
  test('splits picked from unpicked', () => {
    const r = readAnswer('Fixed the merge bug, Run make eval next', pts)
    expect(r.understood).toEqual(['Fixed the merge bug', 'Run make eval next'])
    expect(r.notUnderstood).toEqual(['ρ ≤ 0.80 test drops 12 tasks'])
    expect(r.other).toBe('')
  })
  test('keeps typed text, commas and all', () => {
    const r = readAnswer('Fixed the merge bug, what is rho, and why 0.80', pts)
    expect(r.understood).toEqual(['Fixed the merge bug'])
    expect(r.other).toBe('what is rho, and why 0.80')
  })
  test('an empty submit counts as none', () => {
    expect(readAnswer('  ', pts).none).toBe(true)
  })
  test('none marks every point not understood', () => {
    const r = readAnswer(NONE, pts)
    expect(r.none).toBe(true)
    expect(r.notUnderstood).toEqual(pts)
  })
})
