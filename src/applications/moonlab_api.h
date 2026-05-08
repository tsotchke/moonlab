/**
 * @file moonlab_api.h
 * @brief Standalone definition of the MOONLAB_API visibility macro.
 *
 * Module headers (ca_mps.h, dmrg.h, ...) include THIS rather than
 * moonlab_export.h to pick up the MOONLAB_API tag without dragging
 * in the full ABI surface declaration.  moonlab_export.h itself
 * also includes this file, so the macro is defined exactly once
 * regardless of include order.
 *
 * The macro expansion contract is identical to moonlab_export.h:
 *   - GCC/Clang: __attribute__((visibility("default")))
 *   - MSVC:      __declspec(dllimport / dllexport)
 *   - other:     empty
 *
 * In v0.2.x default-OFF visibility means the tag is observable only
 * when QSIM_HIDDEN_VISIBILITY=ON; v0.3 flips that default and the
 * tag becomes load-bearing.
 *
 * @since v0.2.3
 */
#ifndef MOONLAB_API_H
#define MOONLAB_API_H

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(MOONLAB_BUILDING_SHARED)
#    define MOONLAB_API __declspec(dllexport)
#  elif defined(MOONLAB_USING_SHARED)
#    define MOONLAB_API __declspec(dllimport)
#  else
#    define MOONLAB_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define MOONLAB_API __attribute__((visibility("default")))
#else
#  define MOONLAB_API
#endif

#endif /* MOONLAB_API_H */
