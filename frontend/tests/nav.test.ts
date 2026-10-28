import assert from 'node:assert/strict'
import { test } from 'node:test'
import { safeNext } from '../src/lib/nav.ts'

test('safeNext only allows same-site paths', () => {
  assert.equal(safeNext('/projects/abc?x=1'), '/projects/abc?x=1')
  assert.equal(safeNext(['/settings', '/x']), '/settings')
  assert.equal(safeNext(undefined), '/')
  assert.equal(safeNext(null), '/')
  assert.equal(safeNext('https://evil.example/'), '/')
  assert.equal(safeNext('//evil.example/'), '/')
  assert.equal(safeNext('/\\evil.example'), '/')
  assert.equal(safeNext('/login?next=/x'), '/')
  assert.equal(safeNext('/loginhelp'), '/loginhelp')
})
