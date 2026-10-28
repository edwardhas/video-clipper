import assert from 'node:assert/strict'
import { test } from 'node:test'
import { ACCEPT_ATTR, fileExtension, fileProblem } from '../src/lib/files.ts'

const MB = 1024 * 1024

test('fileExtension is case-insensitive and ignores dotfiles', () => {
  assert.equal(fileExtension('Talk.MP4'), 'mp4')
  assert.equal(fileExtension('.hidden'), '')
  assert.equal(fileExtension('noext'), '')
})

test('accept attribute lists video types', () => {
  assert.ok(ACCEPT_ATTR.startsWith('video/*,'))
  assert.ok(ACCEPT_ATTR.includes('.mkv'))
})

test('fileProblem checks type, emptiness, upload limit and disk space', () => {
  const limits = { maxUploadMb: 100, usableBytes: 500 * MB }
  assert.equal(fileProblem({ name: 'a.mp4', size: 10 * MB }, limits), null)
  assert.match(fileProblem({ name: 'a.pdf', size: 10 }, limits) ?? '', /not supported/)
  assert.match(fileProblem({ name: 'a.mov', size: 0 }, limits) ?? '', /empty/)
  assert.match(fileProblem({ name: 'a.mov', size: 101 * MB }, limits) ?? '', /larger than the 100 MB upload limit/)
  assert.match(
    fileProblem({ name: 'a.mov', size: 90 * MB }, { maxUploadMb: null, usableBytes: 50 * MB }) ?? '',
    /not enough free disk space/,
  )
  assert.equal(fileProblem({ name: 'a.webm', size: 90 * MB }, { maxUploadMb: null, usableBytes: null }), null)
})
