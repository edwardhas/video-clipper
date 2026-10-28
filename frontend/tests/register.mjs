// Lets Node's test runner load the app's TypeScript sources directly:
// Node strips the types; this hook resolves extensionless relative imports to .ts files.
import { register } from 'node:module'

register('./ts-resolve.mjs', import.meta.url)
