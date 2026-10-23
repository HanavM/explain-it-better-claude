import type { EngineInterface, Register } from 'claude-code'

import {
  DEFAULTS,
  FRESH,
  NONE,
  SPLIT_SYSTEM,
  isExplanation,
  jargonScore,
  parsePoints,
  readAnswer,
  reserve,
  rollDay,
  settle,
  shouldAsk,
  type Reading,
  type Throttle,
} from './policy'

// One explanation the agent gave, as written to disk. Every eligible one is
// logged, asked about or not, so later labelling can use the unasked ones too.
type Explanation = {
  id: string
  parentId?: string
  at: number
  session: string
  cwd: string
  origin: string
  prompt: string
  answer: string
  toolCalls: number
  durationMs: number
  jargon: number
  eligible: boolean
  trigger: 'auto' | 'manual' | null
  points: string[]
  skipped: boolean
  feedback: Reading | null
}

type Pending = { record: Explanation; notBefore: number; promptSeq: number }

const THROTTLE_KEY = 'throttle'
const STATS_KEY = 'stats'
// How long an automatic check waits for the prompt box to be empty.
const GIVE_UP_MS = 120_000
type Stats = { asked: number; answered: number; skipped: number }

function roll(): number {
  const buf = new Uint32Array(1)
  crypto.getRandomValues(buf)
  return (buf[0] ?? 0) / 2 ** 32
}

async function loadThrottle($: EngineInterface): Promise<Throttle> {
  const saved = (await $.store.get(THROTTLE_KEY)) as Partial<Throttle> | undefined
  return { ...FRESH, ...saved }
}

async function bumpStats($: EngineInterface, key: keyof Stats): Promise<void> {
  const saved = (await $.store.get(STATS_KEY)) as Partial<Stats> | undefined
  const stats = { asked: 0, answered: 0, skipped: 0, ...saved }
  stats[key] += 1
  await $.store.set(STATS_KEY, stats)
}

async function dataDir($: EngineInterface): Promise<string> {
  const home = (await $.env.get('HOME')) ?? '.'
  return `${home}/explainer-rl/data/events`
}

async function save($: EngineInterface, record: Explanation): Promise<void> {
  await $.fs.write(`${await dataDir($)}/${record.id}.json`, JSON.stringify(record, null, 2) + '\n')
}

// Logs an explanation whose check never showed, as an unasked record.
async function saveUnasked($: EngineInterface, record: Explanation): Promise<void> {
  if (record.eligible || record.trigger === 'manual') await save($, { ...record, trigger: null, points: [] })
}

async function splitPoints($: EngineInterface, answer: string): Promise<string[]> {
  const r = await $.model.complete({
    model: 'haiku',
    system: SPLIT_SYSTEM,
    prompt: `<message>\n${answer.slice(0, 12_000)}\n</message>`,
    maxTokens: 300,
    effort: 'low',
    timeoutMs: 15_000,
  })
  if (!r.isAnswered) $.ui.log(`explain-check: split fell back (${r.reason})`, { to: 'debug' })
  return parsePoints(r.isAnswered ? r.text : undefined, answer)
}

// The live explanation prompt the training loop deployed, as plain text.
async function livePrompt($: EngineInterface): Promise<string> {
  const home = (await $.env.get('HOME')) ?? '.'
  try {
    const cur = JSON.parse(await $.fs.read(`${home}/explainer-rl/prompts/current.json`)) as { name?: string }
    if (!cur.name || !/^v\d{3}$/.test(cur.name)) return ''
    return (await $.fs.read(`${home}/explainer-rl/prompts/${cur.name}.md`)).trim()
  } catch {
    return ''
  }
}

// The module's own state: reset when the mod reloads, which only drops an
// in-flight check. Throttle and stats live in `$.store` and survive.
let isInteractive = false
let isHumanTurn = false
let origin = ''
let toolCalls = 0
let lastPrompt = ''
let last: Explanation | null = null
let pending: Pending | null = null
let isAsking = false
let promptSeq = 0

async function replacePending($: EngineInterface, next: Pending | null): Promise<void> {
  const old = pending
  pending = next
  if (old !== null) await saveUnasked($, old.record)
}

async function drain($: EngineInterface): Promise<void> {
  if (isAsking || pending === null) return
  const now = await $.clock.now()
  if (now < pending.notBefore) return
  const current = pending
  const { record } = current

  if (record.trigger === 'auto') {
    if (promptSeq !== current.promptSeq || now - record.at > GIVE_UP_MS) {
      await replacePending($, null)
      return
    }
    // Never open over a half-typed prompt: try again in a few seconds.
    if ((await $.prompt.read()).text.trim() !== '') {
      current.notBefore = now + 5000
      return
    }
  }

  pending = null
  isAsking = true
  try {
    record.points = await splitPoints($, record.answer)
    // The person may have sent a new prompt or started typing while the
    // split ran; an automatic check would then land on top of that.
    const isStale =
      record.trigger === 'auto' &&
      (promptSeq !== current.promptSeq || (await $.prompt.read()).text.trim() !== '')
    if (record.points.length === 0 || isStale) {
      await saveUnasked($, record)
      return
    }

    await bumpStats($, 'asked')
    let answer: string | undefined
    try {
      answer = await $.ui.ask(
        'Which of these from the last message did you actually understand? (Space ticks, Enter submits, Esc skips. Type under Other to say what was unclear.)',
        { options: [...record.points, NONE], header: 'Understood?', multiSelect: true },
      )
    } catch {
      answer = undefined
    }

    record.skipped = answer === undefined
    record.feedback = answer === undefined ? null : readAnswer(answer, record.points)
    await bumpStats($, record.skipped ? 'skipped' : 'answered')
    if (record.trigger === 'auto') {
      await $.store.set(THROTTLE_KEY, settle(await loadThrottle($), record.skipped))
    }
    await save($, record)
  } catch (err) {
    $.ui.log(`explain-check: ${String(err)}`, { to: 'debug' })
  } finally {
    isAsking = false
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    isInteractive = e.isInteractive
    await $.command.register({
      name: 'explain-check',
      description: 'Rate the last explanation now; or `off`, `on`, `stats`',
    })
    // Headless runs have no one to ask; they would only burn the daily budget.
    if (isInteractive) $.clock.every(1000, () => void drain($))
    return next(e)
  })

  on('prompt.submit', async ($, e, next) => {
    toolCalls = 0
    promptSeq += 1
    origin = e.origin.kind
    // Task notifications, /loop wakeups, peer messages and the like start
    // turns too; only a person's own prompt makes a turn worth asking about.
    isHumanTurn = origin === 'composer' || origin === 'bridge'
    if (isHumanTurn && !e.text.trimStart().startsWith('/')) lastPrompt = e.text
    return next(e)
  })

  // Adds the deployed prompt to the system prompt of interactive sessions
  // only: the pipeline's own headless Claude calls must not see it.
  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    if (!isInteractive) return composed
    const text = await livePrompt($)
    if (text === '') return composed
    return {
      sections: [
        ...composed.sections,
        { id: 'explain-check:explaining', text: `# Explaining your work to this user\n\n${text}`, scope: 'session' as const },
      ],
    }
  })

  on('tool.call', ($, e, next) => {
    toolCalls += 1
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    if (!isInteractive || e.agentId !== undefined || e.reason !== 'answer' || e.answer.trim() === '') return next(e)

    const now = await $.clock.now()
    const eligible = isExplanation(e.answer, toolCalls)
    const jargon = jargonScore(e.answer)
    const record: Explanation = {
      id: `${now}-${crypto.randomUUID().slice(0, 8)}`,
      at: now,
      session: await $.session.id(),
      cwd: await $.session.cwd(),
      origin,
      prompt: isHumanTurn ? lastPrompt.slice(0, 4000) : '',
      answer: e.answer,
      toolCalls,
      durationMs: e.durationMs,
      jargon,
      eligible,
      trigger: null,
      points: [],
      skipped: false,
      feedback: null,
    }
    last = record
    if (!eligible) return next(e)

    const t = rollDay(await loadThrottle($), now)
    const canAsk = isHumanTurn && !isAsking && pending === null
    if (canAsk && shouldAsk(now, t, jargon, roll())) {
      await $.store.set(THROTTLE_KEY, reserve(t, now))
      // A short wait, so a prompt the person fires straight back cancels it.
      pending = { record: { ...record, trigger: 'auto' }, notBefore: now + 2500, promptSeq }
    } else {
      await save($, record)
    }
    return next(e)
  })

  on('command.run', { command: 'explain-check' }, async ($, e) => {
    const arg = e.args.trim().toLowerCase()

    if (arg === 'off' || arg === 'on') {
      const t = await loadThrottle($)
      await $.store.set(THROTTLE_KEY, { ...t, paused: arg === 'off' })
      return { text: arg === 'off' ? 'Automatic explanation checks paused. `/explain-check on` resumes them.' : 'Automatic explanation checks resumed.' }
    }

    if (arg === 'stats') {
      const s = ((await $.store.get(STATS_KEY)) ?? { asked: 0, answered: 0, skipped: 0 }) as Stats
      const t = rollDay(await loadThrottle($), await $.clock.now())
      return {
        text: `Asked ${s.asked}, answered ${s.answered}, skipped ${s.skipped}. Today: ${t.asksToday}/${DEFAULTS.maxPerDay}${t.paused ? ' (paused)' : ''}. Data: ${await dataDir($)}`,
      }
    }

    if (last === null) return { text: 'No agent message in this session yet.' }
    if (isAsking) return { text: 'A check is already open.' }
    const now = await $.clock.now()
    const record: Explanation = { ...last, id: `${last.id}-m${now}`, parentId: last.id, trigger: 'manual' }
    await replacePending($, { record, notBefore: 0, promptSeq })
    return { text: 'Opening the check for the last message.' }
  })
}
