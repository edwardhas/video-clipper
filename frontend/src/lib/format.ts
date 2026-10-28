const pad2 = (n: number) => String(n).padStart(2, '0')

/** mm:ss (or h:mm:ss) for a timeline position in seconds. */
export function fmtClock(sec: number): string {
  const s = Math.max(0, Math.floor(sec + 1e-6))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  return h > 0 ? `${h}:${pad2(m)}:${pad2(r)}` : `${pad2(m)}:${pad2(r)}`
}

/** mm:ss.s (or h:mm:ss.s) with tenths, for trimming. */
export function fmtPrecise(sec: number): string {
  const tenths = Math.max(0, Math.round(sec * 10))
  const whole = Math.floor(tenths / 10)
  const h = Math.floor(whole / 3600)
  const m = Math.floor((whole % 3600) / 60)
  const r = whole % 60
  const t = tenths % 10
  const base = h > 0 ? `${h}:${pad2(m)}:${pad2(r)}` : `${pad2(m)}:${pad2(r)}`
  return `${base}.${t}`
}

/** Source range as "mm:ss–mm:ss". */
export function fmtRange(srcStart: number, srcEnd: number): string {
  return `${fmtClock(srcStart)}–${fmtClock(srcEnd)}`
}

/** Human duration: "42 s", "3 min 05 s", "1 h 04 min". */
export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return '—'
  if (sec < 60) return `${sec < 10 ? sec.toFixed(1).replace(/\.0$/, '') : Math.round(sec)} s`
  const s = Math.round(sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h > 0) return `${h} h ${pad2(m)} min`
  return r ? `${m} min ${pad2(r)} s` : `${m} min`
}

/** Byte sizes in binary units (1 GB = 1024³ bytes), matching how the server's limits are set. */
export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '—'
  const units = ['bytes', 'KB', 'MB', 'GB', 'TB']
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  if (i === 0) return `${Math.round(v)} bytes`
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

/** Upload limits are configured in MiB (1024 × 1024 bytes); show them the same way. */
export function fmtMegabytes(mb: number): string {
  if (mb >= 1024) {
    const gb = mb / 1024
    return `${Number.isInteger(gb) ? gb : gb.toFixed(1)} GB`
  }
  return `${mb} MB`
}

export function mbToBytes(mb: number): number {
  return mb * 1024 * 1024
}

/** Minutes limit in words: "180 minutes" or "3 hours". */
export function fmtMinutes(minutes: number): string {
  if (minutes >= 60 && minutes % 60 === 0) return plural(minutes / 60, 'hour')
  return plural(Math.round(minutes), 'minute')
}

const dateFmt = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

/** Absolute local date/time for a unix timestamp in seconds. */
export function fmtDateTime(unixSeconds: number): string {
  return dateFmt.format(new Date(unixSeconds * 1000))
}

const relFmt = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

/** "5 minutes ago", "yesterday" … for a unix timestamp in seconds. */
export function fmtRelative(unixSeconds: number, nowMs = Date.now()): string {
  const diff = unixSeconds - nowMs / 1000
  const abs = Math.abs(diff)
  if (abs < 45) return 'just now'
  if (abs < 3600) return relFmt.format(Math.round(diff / 60), 'minute')
  if (abs < 86400) return relFmt.format(Math.round(diff / 3600), 'hour')
  if (abs < 86400 * 30) return relFmt.format(Math.round(diff / 86400), 'day')
  return fmtDateTime(unixSeconds)
}

export function fmtScore(score: number): string {
  return `${Math.round(Math.max(0, Math.min(100, score)))}/100`
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`
}
