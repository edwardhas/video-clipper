export async function resolve(specifier, context, next) {
  try {
    return await next(specifier, context)
  } catch (err) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]s(\?.*)?$/.test(specifier)) {
      return next(`${specifier}.ts`, context)
    }
    throw err
  }
}
