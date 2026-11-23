import type { EngineInterface, Register } from 'claude-code'

import {
  DEFAULTS,
  EDIT_TOOLS,
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
  // Which app the session ran in: cli, claude-vscode, claude-desktop...
  app: string
  origin: string
  prompt: string
  answer: string
  toolCalls: number
  edits: number
  durationMs: number
  jargon: number
  eligible: boolean
  trigger: 'auto' | 'manual' | null
  points: string[]
  skipped: boolean
  feedback: Reading | null
  // The explanation instructions version the session was composed with, so
  // an A/B test can credit this message to the right version.
  promptVersion?: string
  // Which check was shown: the understanding check (which points landed) or
  // the quick 1-5 rating. Absent on records from before ratings existed.
  kind?: 'understood' | 'rating'
  // The 1-5 answer to "How much did you like the explanation?"; null if skipped.
  rating?: number | null
}

type Pending = { record: Explanation; notBefore: number; promptSeq: number }

const THROTTLE_KEY = 'throttle'
// Share of automatic checks that are the quick rating instead of the understanding check.
const RATING_SHARE = 0.5
// An open rating row goes away unanswered after this long.
const RATING_TIMEOUT_MS = 5 * 60_000
const RATINGS = ['Not at all', 'A little', 'Somewhat', 'A lot', 'Loved it'] as const
const STATS_KEY = 'stats'
// How long an automatic check waits for the prompt box to be empty.
const GIVE_UP_MS = 120_000
// Apps where a person is at the prompt even though Claude Code runs there as
// an SDK host (isInteractive false, prompts arrive as origin 'sdk').
const PERSON_APPS = new Set(['claude-vscode', 'claude-desktop'])
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

async function splitPoints($: EngineInterface, answer: string, request: string, isManual: boolean): Promise<string[]> {
  const r = await $.model.complete({
    model: 'haiku',
    system: SPLIT_SYSTEM,
    prompt: `<request>\n${request.slice(0, 3000) || '(not available)'}\n</request>\n\n<message>\n${answer.slice(0, 12_000)}\n</message>`,
    maxTokens: 300,
    effort: 'low',
    timeoutMs: 15_000,
  })
  if (!r.isAnswered) $.ui.log(`explain-check: split fell back (${r.reason})`, { to: 'debug' })
  return parsePoints(r.isAnswered ? r.text : undefined, answer, isManual)
}

// The live explanation prompt the training loop deployed: its version name and text.
async function livePrompt($: EngineInterface): Promise<{ name: string; text: string }> {
  const home = (await $.env.get('HOME')) ?? '.'
  try {
    const cur = JSON.parse(await $.fs.read(`${home}/explainer-rl/prompts/current.json`)) as { name?: string }
    if (!cur.name || !/^v\d{3}$/.test(cur.name)) return { name: '', text: '' }
    return { name: cur.name, text: (await $.fs.read(`${home}/explainer-rl/prompts/${cur.name}.md`)).trim() }
  } catch {
    return { name: '', text: '' }
  }
}

// The module's own state: reset when the mod reloads, which only drops an
// in-flight check. Throttle and stats live in `$.store` and survive.
let isInteractive = false
let app = ''
let isHumanTurn = false
let origin = ''
let toolCalls = 0
let edits = 0
let lastPrompt = ''
let last: Explanation | null = null
let pending: Pending | null = null
let isAsking = false
let promptSeq = 0
let composedVersion = ''
// The rating row above the prompt, while one is open.
let rating: { record: Explanation; openedAt: number; promptSeq: number } | null = null

// The prompt box's text, or '' on a host that has no box to read.
async function typedText($: EngineInterface): Promise<string> {
  try {
    return (await $.prompt.read()).text.trim()
  } catch {
    return ''
  }
}

async function replacePending($: EngineInterface, next: Pending | null): Promise<void> {
  const old = pending
  pending = next
  if (old !== null) await saveUnasked($, old.record)
}

async function drain($: EngineInterface): Promise<void> {
  if (rating !== null && (await $.clock.now()) - rating.openedAt > RATING_TIMEOUT_MS) await closeRating($, null)
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
    if ((await typedText($)) !== '') {
      current.notBefore = now + 5000
      return
    }
  }

  pending = null
  isAsking = true
  if (record.trigger === 'auto' && roll() < RATING_SHARE) {
    record.kind = 'rating'
    rating = { record, openedAt: now, promptSeq: current.promptSeq }
    $.ui.invalidate('ui.render')
    return // isAsking stays true until the row closes
  }
  record.kind = 'understood'
  try {
    record.points = await splitPoints($, record.answer, record.prompt, record.trigger === 'manual')
    // The person may have sent a new prompt or started typing while the
    // split ran; an automatic check would then land on top of that.
    const isStale =
      record.trigger === 'auto' &&
      (promptSeq !== current.promptSeq || (await typedText($)) !== '')
    if (record.points.length === 0 || isStale) {
      await saveUnasked($, record)
      return
    }

    let answer: string | undefined
    try {
      answer = await $.ui.ask(
        'Which parts of the last message could you actually follow? (Space ticks, Enter submits, Esc skips.)',
        { options: [...record.points, NONE], header: 'Understood?', multiSelect: true },
      )
    } catch (err) {
      // Esc rejects too, but so does a dialog the engine refuses: log it so a
      // refusal is not silently filed as a dismissal again.
      $.ui.log(`explain-check: ask rejected (${String(err)})`, { to: 'debug' })
      answer = undefined
    }

    await bumpStats($, 'asked')
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

// Closes the rating row: a press saves the score, anything else a skip.
async function closeRating($: EngineInterface, score: number | null): Promise<void> {
  const open = rating
  if (open === null) return
  rating = null
  isAsking = false
  $.ui.invalidate('ui.render')
  const record = { ...open.record, rating: score, skipped: score === null }
  await bumpStats($, 'asked')
  await bumpStats($, record.skipped ? 'skipped' : 'answered')
  if (record.trigger === 'auto') await $.store.set(THROTTLE_KEY, settle(await loadThrottle($), record.skipped))
  await save($, record)
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    app = (await $.env.get('CLAUDE_CODE_ENTRYPOINT')) ?? ''
    // The pipeline's own `claude -p` calls set EXPLAIN_CHECK_OFF, so they get
    // neither checks nor the deployed prompt whatever app launched them.
    const isOff = ((await $.env.get('EXPLAIN_CHECK_OFF')) ?? '') !== ''
    isInteractive = !isOff && (e.isInteractive || PERSON_APPS.has(app))
    await $.command.register({
      name: 'explain-check',
      description: 'Check the last explanation now; `rate` for the quick 1-5 rating; or `off`, `on`, `stats`',
    })
    // Headless runs (-p, the pipeline) have no one to ask; they would only burn
    // the daily budget. VS Code and the desktop app count as interactive.
    if (isInteractive) $.clock.every(1000, () => void drain($))
    return next(e)
  })

  on('prompt.submit', async ($, e, next) => {
    // Sending the next prompt instead of rating counts as a skip.
    if (rating !== null) await closeRating($, null)
    toolCalls = 0
    edits = 0
    promptSeq += 1
    origin = e.origin.kind
    // Task notifications, /loop wakeups, peer messages and the like start
    // turns too; only a person's own prompt makes a turn worth asking about.
    // In VS Code and the desktop app the person's own prompt arrives as 'sdk'.
    isHumanTurn = origin === 'composer' || origin === 'bridge' || (origin === 'sdk' && PERSON_APPS.has(app))
    if (isHumanTurn && !e.text.trimStart().startsWith('/')) lastPrompt = e.text
    return next(e)
  })

  // Adds the deployed prompt to the system prompt of interactive sessions
  // only: the pipeline's own headless Claude calls must not see it.
  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    if (!isInteractive) return composed
    const { name, text } = await livePrompt($)
    composedVersion = name
    if (text === '') return composed
    return {
      sections: [
        ...composed.sections,
        { id: 'explain-check:explaining', text: `# Explaining your work to this user\n\n${text}`, scope: 'session' as const },
      ],
    }
  })

  // The quick rating: a row above the prompt, answered by a bare 1-5 in an
  // empty prompt box (or a click), the way Claude Code's own survey is.
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const open = rating
    if (open === null || e.props.hasSurvey) return next(e)
    const { Box, Button, Text } = $.ui.resolve(e)
    return (
      <Box flexDirection="column">
        <Text>How much did you like that explanation? <Text dimColor>(press 1-5 · 0 skips)</Text></Text>
        <Box gap={2} flexWrap="wrap">
          {RATINGS.map((label, i) => (
            <Button key={`rate-${i + 1}`} hotkey={String(i + 1)} plain label={label} onPress={() => closeRating($, i + 1)} />
          ))}
          <Button key="rate-skip" hotkey="0" plain dimColor label="Skip" onPress={() => closeRating($, null)} />
        </Box>
      </Box>
    )
  })

  on('tool.call', ($, e, next) => {
    toolCalls += 1
    if (EDIT_TOOLS.has(e.tool)) edits += 1
    return next(e)
  })

  on('turn.complete', async ($, e, next) => {
    if (!isInteractive || e.agentId !== undefined || e.reason !== 'answer' || e.answer.trim() === '') return next(e)

    const now = await $.clock.now()
    const eligible = isExplanation(e.answer)
    const jargon = jargonScore(e.answer)
    const record: Explanation = {
      id: `${now}-${crypto.randomUUID().slice(0, 8)}`,
      at: now,
      session: await $.session.id(),
      cwd: await $.session.cwd(),
      app,
      origin,
      prompt: isHumanTurn ? lastPrompt.slice(0, 4000) : '',
      answer: e.answer,
      toolCalls,
      edits,
      durationMs: e.durationMs,
      jargon,
      eligible,
      trigger: null,
      points: [],
      skipped: false,
      feedback: null,
      promptVersion: composedVersion || undefined,
    }
    last = record
    if (!eligible) return next(e)

    const t = rollDay(await loadThrottle($), now)
    const canAsk = isHumanTurn && !isAsking && pending === null
    if (canAsk && shouldAsk(now, t, jargon, toolCalls > 0, roll())) {
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
    if (arg === 'rate') {
      const record: Explanation = { ...last, id: `${last.id}-m${now}`, parentId: last.id, trigger: 'manual', kind: 'rating' }
      isAsking = true
      rating = { record, openedAt: now, promptSeq }
      $.ui.invalidate('ui.render')
      return { text: 'Rate the last message: press 1-5 with the prompt box empty (0 skips).' }
    }
    const record: Explanation = { ...last, id: `${last.id}-m${now}`, parentId: last.id, trigger: 'manual' }
    await replacePending($, { record, notBefore: 0, promptSeq })
    return { text: 'Opening the check for the last message.' }
  })
}
