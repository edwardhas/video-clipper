import assert from 'node:assert/strict'
import { test } from 'node:test'
import { youtubeUrlProblem } from '../src/lib/youtube.ts'

test('accepts single-video YouTube links in common shapes', () => {
  for (const url of [
    'https://www.youtube.com/watch?v=nKyWusznRgQ',
    'youtube.com/watch?v=nKyWusznRgQ&t=30s',
    'https://youtu.be/nKyWusznRgQ',
    'https://m.youtube.com/watch?v=nKyWusznRgQ',
    'https://music.youtube.com/watch?v=nKyWusznRgQ',
    'https://www.youtube.com/shorts/abc123',
    'https://www.youtube.com/live/abc123',
    'https://www.youtube.com/embed/abc123',
  ]) {
    assert.equal(youtubeUrlProblem(url), null, url)
  }
})

test('rejects empty input, other hosts, playlists and channels with a reason', () => {
  assert.match(youtubeUrlProblem('   ') ?? '', /Paste a YouTube link/)
  assert.match(youtubeUrlProblem('https://vimeo.com/123') ?? '', /Only YouTube links/)
  assert.match(youtubeUrlProblem('https://www.youtube.com/playlist?list=PL1') ?? '', /single video/)
  assert.match(youtubeUrlProblem('https://www.youtube.com/@channel') ?? '', /single video/)
  assert.match(youtubeUrlProblem('http://') ?? '', /does not look like a link/)
})

// Finding 43: the host check must match backend/clipper/media/youtube.py parse_youtube_url
// (_VIDEO_HOSTS, the exact `youtu.be` short host, http(s) only, and its schemeless shapes).
test('accepts exactly the hosts the server accepts', () => {
  for (const url of [
    'https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ',
    'http://youtu.be/dQw4w9WgXcQ',
    'HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ',
    'youtu.be/dQw4w9WgXcQ',
    'www.youtube.com/watch?v=dQw4w9WgXcQ',
    'm.youtube.com/watch?v=dQw4w9WgXcQ',
    'music.youtube.com/watch?v=dQw4w9WgXcQ',
  ]) {
    assert.equal(youtubeUrlProblem(url), null, url)
  }
})

test('rejects hosts and shapes the server rejects', () => {
  assert.match(youtubeUrlProblem('https://www.youtu.be/dQw4w9WgXcQ') ?? '', /Only YouTube links/)
  assert.match(youtubeUrlProblem('https://notyoutu.be/dQw4w9WgXcQ') ?? '', /Only YouTube links/)
  assert.match(youtubeUrlProblem('https://youtube-nocookie.com/embed/dQw4w9WgXcQ') ?? '', /Only YouTube links/)
  assert.match(youtubeUrlProblem('ftp://www.youtube.com/watch?v=dQw4w9WgXcQ') ?? '', /http/)
  // the server only adds https:// to schemeless youtube.com / youtu.be links
  assert.match(youtubeUrlProblem('www.youtube-nocookie.com/embed/dQw4w9WgXcQ') ?? '', /https:\/\//)
})
