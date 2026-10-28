// Mirrors the authoritative parser, backend/clipper/media/youtube.py parse_youtube_url:
// the same hosts (_VIDEO_HOSTS + the exact _SHORT_HOST), http(s) only, and the same
// schemeless shapes it completes with https:// (_SCHEMELESS_RE).
const SHORT_HOST = 'youtu.be'
const VIDEO_HOSTS = new Set([
  'youtube.com',
  'www.youtube.com',
  'm.youtube.com',
  'music.youtube.com',
  'www.youtube-nocookie.com',
])
const SCHEMELESS = /^(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be)\//i
const HAS_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i

const NOT_YOUTUBE = 'Only YouTube links are supported. For other videos, download the file and upload it instead.'

function isYoutubeHost(host: string): boolean {
  return host === SHORT_HOST || VIDEO_HOSTS.has(host)
}

/**
 * Quick client-side sanity check so obvious typos are caught before submitting.
 * The server performs the authoritative parsing.
 */
export function youtubeUrlProblem(raw: string): string | null {
  const value = raw.trim()
  if (!value) return 'Paste a YouTube link.'
  const schemeless = !HAS_SCHEME.test(value)
  let url: URL
  try {
    url = new URL(schemeless ? `https://${value}` : value)
  } catch {
    return 'This does not look like a link. Paste the full YouTube address.'
  }
  const host = url.hostname.toLowerCase()
  if (schemeless && !SCHEMELESS.test(value)) {
    // The server only completes youtube.com / youtu.be links without a scheme.
    return isYoutubeHost(host) ? 'Paste the full link, starting with https://.' : NOT_YOUTUBE
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return 'Only http:// or https:// YouTube links are supported.'
  if (!isYoutubeHost(host) || url.username || url.password || (url.port && url.port !== '80' && url.port !== '443')) {
    return NOT_YOUTUBE
  }
  const path = url.pathname
  const isVideo =
    (host === SHORT_HOST && path.length > 1) ||
    (path === '/watch' && url.searchParams.has('v')) ||
    /^\/(shorts|live|embed)\/[^/]+/.test(path)
  if (!isVideo) return 'Paste a link to a single video (playlists and channels are not supported).'
  return null
}
