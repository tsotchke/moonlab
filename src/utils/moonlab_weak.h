#ifndef MOONLAB_WEAK_H
#define MOONLAB_WEAK_H

/*
 * Optional backend entry points are allowed to be absent from a CPU-only
 * build.  ELF accepts an undefined weak reference, while Mach-O requires a
 * weak import for the same NULL-at-runtime behavior.
 */
#if defined(__APPLE__)
#define MOONLAB_WEAK_IMPORT __attribute__((weak_import))
#else
#define MOONLAB_WEAK_IMPORT __attribute__((weak))
#endif

#endif /* MOONLAB_WEAK_H */
