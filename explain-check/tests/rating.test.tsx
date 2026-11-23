import { expect, test } from 'claude-code/testing'

const BAND = {
  plugin: 'explain-check',
  component: 'AbovePrompt',
  props: { hasSurvey: false, isCollapsed: false, isWorking: false, maxRows: 10, bodyColumns: 80 },
} as const

test('the 1-5 rating row saves the number pressed, and closes', async ($, on) => {
  const writes: { path: string; text: string }[] = []
  on('command.register', () => ({ value: { command: 'explain-check' } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('turn.complete', () => ({ text: '' }))
  on('env.get', (_, e) => ({ value: (e as { name?: string }).name === 'HOME' ? '/home' : undefined }))
  on('clock.every', () => ({ value: undefined }))
  on('session.id', () => ({ value: 's1' }))
  on('clock.now', () => ({ value: 1_800_000_000_000 }))
  on('session.cwd', () => ({ value: '/' }))
  on('store.get', () => ({ value: undefined }))
  on('store.set', () => ({ value: undefined }))
  on('fs.read', () => { throw new Error('no file') })
  on('fs.write', (_, e) => {
    writes.push(e as unknown as { path: string; text: string })
    return { value: undefined }
  })

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true } as never)
  await $.turn.complete({ reason: 'answer', answer: 'A long explanation. '.repeat(40), durationMs: 1 } as never)
  await $.command.run({ command: 'explain-check', args: 'rate' } as never)

  const ui = await $.ui.mount({ ...BAND, surface: 'terminal' } as never)
  expect((await ui.find({ type: 'Text', text: /How much did you like/ }))).toBeDefined()
  await ui.press({ key: 'rate-4' })
  await ui.unmount()

  const saved = writes.map(w => JSON.parse(w.text)).find(r => r.kind === 'rating')
  expect(saved?.rating).toBe(4)
  expect(saved?.skipped).toBe(false)

  const after = await $.ui.mount({ ...BAND, surface: 'terminal' } as never)
  expect(await after.find({ type: 'Text', text: /How much did you like/ })).toBeUndefined()
  await after.unmount()
})
