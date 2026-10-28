type QueryValue = string | null | undefined | (string | null)[]

/** Only same-site paths are accepted as a post-login destination (no open redirects). */
export function safeNext(raw: QueryValue): string {
  const value = Array.isArray(raw) ? raw[0] : raw
  if (typeof value !== 'string') return '/'
  if (!value.startsWith('/') || value.startsWith('//') || value.startsWith('/\\')) return '/'
  if (value === '/login' || value.startsWith('/login?') || value.startsWith('/login/')) return '/'
  return value
}
