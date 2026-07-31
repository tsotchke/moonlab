/**
 * @file stim_dem.c
 * @brief Stim detector-error-model (`.dem`) import and export.
 *
 * IMPORT is a two stage pipeline.  The text is first parsed into an
 * instruction tree, because `repeat` blocks nest and a block has to be
 * replayed once per iteration; the tree is then flattened, resolving
 * `shift_detectors` into absolute detector indices, and every `error`
 * instruction is split on `^` into its graphlike components.  Components
 * become matching-graph edges keyed on
 * (min detector, max detector, observable mask), parallel mechanisms merging
 * as p = p1(1-p2) + p2(1-p1), and every multi-component mechanism contributes
 * the pairwise correlation links the two-pass decoder consumes.  That is the
 * same model PyMatching builds internally, plus the correlation Stim's
 * decomposition recorded and a matching decoder throws away.
 *
 * The edge ordering is load bearing: edges are indexed by FIRST APPEARANCE in
 * the flattened instruction stream, because correlation links are pairs of
 * edge indices.  That matches the reference converter in
 * benchmarks/dominance/fronts/f3_decoder_vs_pymatching.py::dem_to_edges, so
 * the two produce identical arrays for the same input.
 *
 * EXPORT comes in two flavours, and they are deliberately different.
 *
 * - moonlab_dem_to_text() replays the flattened mechanism list verbatim,
 *   including non-graphlike (>2 detector) mechanisms and observable-only
 *   ones.  Nothing is lost, so parse -> to_text -> parse is an exact fixed
 *   point on the whole model rather than only on its matching-graph
 *   projection.  Mechanisms whose probability lies outside (0, 1) are the one
 *   exception: they never fire, or always fire, and are dropped on import
 *   exactly as the reference converter drops them.
 *
 * - moonlab_dem_text_from_edges() has only the merged edge list and the
 *   pairwise links to work from, so it inverts the merge: each link (u, v, q)
 *   becomes one decomposed mechanism `error(q) <u> ^ <v>`, and each edge
 *   carries what is left after peeling every joint probability touching it
 *   off its total, p_resid = (p - q) / (1 - 2q) applied once per link.
 *   Peeling is exact when every mechanism had at most two components; a
 *   mechanism with three or more contributes to several links at once, so its
 *   probability would be subtracted more than once.  That drives a residual
 *   negative and the call fails with MOONLAB_STIM_ERR_BAD_ARG rather than
 *   clamping to a model the caller did not ask for.  Callers holding a parsed
 *   model should use moonlab_dem_to_text(), which has no such limit.
 *
 * The emission order of moonlab_dem_text_from_edges() is chosen so that
 * re-parsing recovers the ORIGINAL edge indices: an edge is introduced by the
 * first instruction that mentions it, so instructions are ordered to
 * introduce edge 0, then edge 1, and so on.  An edge with no residual is
 * introduced by a link to an already introduced partner whenever one exists,
 * which is what a Stim decomposition always provides.
 */
#include "stim_dem.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Residuals at or below this are taken as fully explained by the
 *  correlation links, and no residual instruction is emitted. */
#define DEM_RESID_SUPPRESS 1e-15
/** How far below zero a residual may drift on rounding before the caller's
 *  inputs are declared inconsistent. */
#define DEM_RESID_TOL 1e-12
/** Bound on `repeat` nesting, so a pathological file cannot exhaust the
 *  stack in the parser or the flattener. */
#define DEM_MAX_NEST 64

/* ================================================================== */
/*  Small utilities                                                    */
/* ================================================================== */

static void dem_err_clear(moonlab_stim_error_t* err) {
    if (!err) return;
    err->code = MOONLAB_STIM_OK;
    err->line = 0;
    err->message[0] = '\0';
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 4, 5)))
#endif
static void dem_err_set(moonlab_stim_error_t* err, int code, size_t line,
                        const char* fmt, ...) {
    if (!err) return;
    err->code = code;
    err->line = line;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err->message, sizeof err->message, fmt, ap);
    va_end(ap);
}

/**
 * @brief Grow a heap array to hold at least @p need elements.
 * @return the new block, or NULL on failure, in which case the old block is
 *         untouched and still owned by the caller.
 */
static void* dem_grow(void* p, size_t* cap, size_t need, size_t esz) {
    if (need <= *cap) return p;
    size_t nc = *cap ? *cap : 16;
    while (nc < need) {
        if (nc > SIZE_MAX / 2) return NULL;
        nc *= 2;
    }
    if (nc > SIZE_MAX / esz) return NULL;
    void* np = realloc(p, nc * esz);
    if (!np) return NULL;
    *cap = nc;
    return np;
}

/* --- string builder ------------------------------------------------ */

typedef struct {
    char*  s;
    size_t len;
    size_t cap;
} dem_sb_t;

static int sb_reserve(dem_sb_t* b, size_t extra) {
    size_t need = b->len + extra + 1;
    if (need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 512;
    while (nc < need) {
        if (nc > SIZE_MAX / 2) return -1;
        nc *= 2;
    }
    char* ns = (char*)realloc(b->s, nc);
    if (!ns) return -1;
    b->s = ns;
    b->cap = nc;
    return 0;
}

static int sb_add(dem_sb_t* b, const char* txt) {
    size_t n = strlen(txt);
    if (sb_reserve(b, n) != 0) return -1;
    memcpy(b->s + b->len, txt, n);
    b->len += n;
    b->s[b->len] = '\0';
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
static int sb_addf(dem_sb_t* b, const char* fmt, ...) {
    char tmp[192];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n < sizeof tmp) return sb_add(b, tmp);
    char* big = (char*)malloc((size_t)n + 1);
    if (!big) return -1;
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    int rc = sb_add(b, big);
    free(big);
    return rc;
}

/* ================================================================== */
/*  Instruction tree                                                   */
/* ================================================================== */

enum {
    DI_ERROR = 0,
    DI_DETECTOR,
    DI_LOGICAL,
    DI_SHIFT,
    DI_REPEAT
};

enum {
    DT_DET = 0, /**< D<uint>   */
    DT_OBS,     /**< L<uint>   */
    DT_NUM,     /**< bare uint */
    DT_SEP      /**< '^'       */
};

typedef struct {
    uint8_t  kind;
    uint64_t val;
} dem_targ_t;

typedef struct dem_block dem_block_t;

typedef struct {
    int          kind;
    size_t       line;
    double*      args;
    size_t       nargs;
    dem_targ_t*  targs;
    size_t       ntargs;
    uint64_t     rep_count;
    dem_block_t* body;
} dem_inst_t;

struct dem_block {
    dem_inst_t* v;
    size_t      n;
    size_t      cap;
};

static void dem_block_free(dem_block_t* b);

static void dem_inst_release(dem_inst_t* in) {
    free(in->args);
    free(in->targs);
    if (in->body) {
        dem_block_free(in->body);
        free(in->body);
    }
}

static void dem_block_free(dem_block_t* b) {
    if (!b) return;
    for (size_t i = 0; i < b->n; i++) dem_inst_release(&b->v[i]);
    free(b->v);
    b->v = NULL;
    b->n = b->cap = 0;
}

/* ================================================================== */
/*  Parser                                                             */
/* ================================================================== */

typedef struct {
    const char*           s;
    size_t                pos;
    size_t                line;
    moonlab_stim_error_t* err;
    int                   failed;
} dem_parser_t;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 4, 5)))
#endif
static void dp_fail(dem_parser_t* p, int code, size_t line,
                    const char* fmt, ...) {
    if (p->failed) return;
    p->failed = 1;
    if (!p->err) return;
    p->err->code = code;
    p->err->line = line;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->err->message, sizeof p->err->message, fmt, ap);
    va_end(ap);
}

static int dp_is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }
static int dp_is_digit(char c) { return c >= '0' && c <= '9'; }
static int dp_is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

/** Copy the token starting at @p pos into @p out for an error message. */
static void dp_snip(const dem_parser_t* p, size_t pos, char* out, size_t cap) {
    if (cap == 0) return;
    char c0 = p->s[pos];
    if (c0 == '\0') { snprintf(out, cap, "end of file"); return; }
    if (c0 == '\n') { snprintf(out, cap, "end of line"); return; }
    size_t i = 0;
    while (i + 1 < cap) {
        char c = p->s[pos + i];
        if (c == '\0' || c == '\n' || dp_is_space(c)) break;
        out[i++] = c;
    }
    if (i == 0) out[i++] = c0;
    out[i] = '\0';
}

/** Whitespace and comments, never crossing a newline. */
static void dp_skip_inline(dem_parser_t* p) {
    for (;;) {
        char c = p->s[p->pos];
        if (dp_is_space(c)) { p->pos++; continue; }
        if (c == '#') {
            while (p->s[p->pos] != '\0' && p->s[p->pos] != '\n') p->pos++;
            continue;
        }
        return;
    }
}

/** Whitespace, comments and newlines. */
static void dp_skip_all(dem_parser_t* p) {
    for (;;) {
        char c = p->s[p->pos];
        if (c == '\n') { p->pos++; p->line++; continue; }
        if (dp_is_space(c)) { p->pos++; continue; }
        if (c == '#') {
            while (p->s[p->pos] != '\0' && p->s[p->pos] != '\n') p->pos++;
            continue;
        }
        return;
    }
}

/** Read an unsigned integer.  0 on success, -1 for no digits, -2 on
 *  overflow. */
static int dp_uint(dem_parser_t* p, uint64_t* out) {
    if (!dp_is_digit(p->s[p->pos])) return -1;
    uint64_t v = 0;
    int over = 0;
    while (dp_is_digit(p->s[p->pos])) {
        uint64_t dg = (uint64_t)(p->s[p->pos] - '0');
        if (v > (UINT64_MAX - dg) / 10u) over = 1;
        if (!over) v = v * 10u + dg;
        p->pos++;
    }
    if (over) return -2;
    *out = v;
    return 0;
}

/** True when the character ends a target list. */
static int dp_ends_targets(char c) {
    return c == '\0' || c == '\n' || c == '{' || c == '}';
}

/** After a target token only a separator may follow. */
static int dp_targ_boundary(char c) {
    return c == '\0' || c == '\n' || c == '#' || c == '{' || c == '}' ||
           dp_is_space(c);
}

static int dp_block(dem_parser_t* p, dem_block_t* blk, int depth);

static int dp_instruction(dem_parser_t* p, dem_block_t* blk, int depth) {
    const size_t name_pos = p->pos;
    const size_t line = p->line;

    char name[64];
    size_t nl = 0;
    while (dp_is_alpha(p->s[p->pos]) || dp_is_digit(p->s[p->pos])) {
        if (nl + 1 < sizeof name) {
            char c = p->s[p->pos];
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            name[nl++] = c;
        }
        p->pos++;
    }
    name[nl] = '\0';

    int kind;
    if (strcmp(name, "error") == 0)                   kind = DI_ERROR;
    else if (strcmp(name, "detector") == 0)           kind = DI_DETECTOR;
    else if (strcmp(name, "logical_observable") == 0) kind = DI_LOGICAL;
    else if (strcmp(name, "shift_detectors") == 0)    kind = DI_SHIFT;
    else if (strcmp(name, "repeat") == 0)             kind = DI_REPEAT;
    else {
        char tok[48];
        dp_snip(p, name_pos, tok, sizeof tok);
        dp_fail(p, MOONLAB_STIM_ERR_UNSUPPORTED, line,
                "unknown detector error model instruction '%s'", tok);
        return -1;
    }

    dem_inst_t in;
    memset(&in, 0, sizeof in);
    in.kind = kind;
    in.line = line;

    size_t args_cap = 0, targs_cap = 0;
    char tok[48];

    /* Optional tag: NAME[...]. */
    if (p->s[p->pos] == '[') {
        size_t tag_pos = p->pos;
        p->pos++;
        while (p->s[p->pos] != ']' && p->s[p->pos] != '\0' &&
               p->s[p->pos] != '\n')
            p->pos++;
        if (p->s[p->pos] != ']') {
            dp_snip(p, tag_pos, tok, sizeof tok);
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "unterminated tag starting at '%s'", tok);
            goto fail;
        }
        p->pos++;
    }

    /* Optional parenthesised arguments. */
    if (p->s[p->pos] == '(') {
        p->pos++;
        dp_skip_inline(p);
        if (p->s[p->pos] == ')') {
            p->pos++;
        } else {
            for (;;) {
                dp_skip_inline(p);
                const char* start = p->s + p->pos;
                char* end = NULL;
                double v = strtod(start, &end);
                if (end == start) {
                    dp_snip(p, p->pos, tok, sizeof tok);
                    dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                            "expected a number in '%s(...)', got '%s'",
                            name, tok);
                    goto fail;
                }
                p->pos += (size_t)(end - start);
                void* np = dem_grow(in.args, &args_cap, in.nargs + 1,
                                    sizeof(double));
                if (!np) {
                    dp_fail(p, MOONLAB_STIM_ERR_OOM, line,
                            "out of memory parsing '%s' arguments", name);
                    goto fail;
                }
                in.args = (double*)np;
                in.args[in.nargs++] = v;
                dp_skip_inline(p);
                if (p->s[p->pos] == ',') { p->pos++; continue; }
                if (p->s[p->pos] == ')') { p->pos++; break; }
                dp_snip(p, p->pos, tok, sizeof tok);
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                        "expected ',' or ')' in '%s(...)', got '%s'",
                        name, tok);
                goto fail;
            }
        }
    }

    /* Targets, up to the end of the line or a brace. */
    for (;;) {
        dp_skip_inline(p);
        char c = p->s[p->pos];
        if (dp_ends_targets(c)) break;

        size_t tpos = p->pos;
        dem_targ_t t;
        memset(&t, 0, sizeof t);

        if (c == '^') {
            p->pos++;
            t.kind = DT_SEP;
        } else if (c == 'D' || c == 'd' || c == 'L' || c == 'l') {
            int is_obs = (c == 'L' || c == 'l');
            p->pos++;
            uint64_t v = 0;
            int rc = dp_uint(p, &v);
            if (rc == -1) {
                dp_snip(p, tpos, tok, sizeof tok);
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                        "expected digits after '%c' in target '%s'",
                        is_obs ? 'L' : 'D', tok);
                goto fail;
            }
            if (rc == -2) {
                dp_snip(p, tpos, tok, sizeof tok);
                dp_fail(p, MOONLAB_STIM_ERR_OVERFLOW, line,
                        "target index in '%s' overflows a 64 bit integer",
                        tok);
                goto fail;
            }
            if (is_obs && v >= 64u) {
                dp_snip(p, tpos, tok, sizeof tok);
                dp_fail(p, MOONLAB_STIM_ERR_UNSUPPORTED, line,
                        "observable target '%s' is out of range: the "
                        "decoder's observable mask is a uint64_t, so indices "
                        "must be below 64", tok);
                goto fail;
            }
            t.kind = (uint8_t)(is_obs ? DT_OBS : DT_DET);
            t.val = v;
        } else if (dp_is_digit(c)) {
            uint64_t v = 0;
            if (dp_uint(p, &v) != 0) {
                dp_snip(p, tpos, tok, sizeof tok);
                dp_fail(p, MOONLAB_STIM_ERR_OVERFLOW, line,
                        "integer target '%s' overflows a 64 bit integer", tok);
                goto fail;
            }
            t.kind = DT_NUM;
            t.val = v;
        } else {
            dp_snip(p, tpos, tok, sizeof tok);
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "unrecognised target '%s' in '%s': expected D<n>, L<n>, "
                    "an integer, or '^'", tok, name);
            goto fail;
        }

        if (!dp_targ_boundary(p->s[p->pos])) {
            dp_snip(p, tpos, tok, sizeof tok);
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "trailing characters in target '%s' of '%s'", tok, name);
            goto fail;
        }

        void* np = dem_grow(in.targs, &targs_cap, in.ntargs + 1,
                            sizeof(dem_targ_t));
        if (!np) {
            dp_fail(p, MOONLAB_STIM_ERR_OOM, line,
                    "out of memory parsing '%s' targets", name);
            goto fail;
        }
        in.targs = (dem_targ_t*)np;
        in.targs[in.ntargs++] = t;
    }

    /* Per-instruction validation. */
    if (kind == DI_ERROR) {
        if (in.nargs != 1) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'error' takes exactly one probability argument, got %zu",
                    in.nargs);
            goto fail;
        }
        if (!(in.args[0] >= 0.0) || in.args[0] > 1.0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'error' probability %.17g is outside [0, 1]", in.args[0]);
            goto fail;
        }
        for (size_t i = 0; i < in.ntargs; i++) {
            if (in.targs[i].kind == DT_NUM) {
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                        "'error' target '%llu' must be D<n>, L<n> or '^'",
                        (unsigned long long)in.targs[i].val);
                goto fail;
            }
        }
    } else if (kind == DI_DETECTOR) {
        if (in.ntargs == 0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'detector' needs at least one D<n> target");
            goto fail;
        }
        for (size_t i = 0; i < in.ntargs; i++) {
            if (in.targs[i].kind != DT_DET) {
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                        "'detector' takes only D<n> targets, got '%s%llu'",
                        in.targs[i].kind == DT_OBS ? "L" :
                        in.targs[i].kind == DT_SEP ? "^" : "",
                        (unsigned long long)in.targs[i].val);
                goto fail;
            }
        }
    } else if (kind == DI_LOGICAL) {
        if (in.nargs != 0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'logical_observable' takes no parenthesised arguments");
            goto fail;
        }
        if (in.ntargs == 0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'logical_observable' needs at least one L<n> target");
            goto fail;
        }
        for (size_t i = 0; i < in.ntargs; i++) {
            if (in.targs[i].kind != DT_OBS) {
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                        "'logical_observable' takes only L<n> targets, got "
                        "'%s%llu'",
                        in.targs[i].kind == DT_DET ? "D" :
                        in.targs[i].kind == DT_SEP ? "^" : "",
                        (unsigned long long)in.targs[i].val);
                goto fail;
            }
        }
    } else if (kind == DI_SHIFT) {
        if (in.ntargs != 1 || in.targs[0].kind != DT_NUM) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'shift_detectors' takes exactly one integer detector "
                    "offset target, got %zu", in.ntargs);
            goto fail;
        }
    } else { /* DI_REPEAT */
        if (in.nargs != 0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'repeat' takes no parenthesised arguments");
            goto fail;
        }
        if (in.ntargs != 1 || in.targs[0].kind != DT_NUM) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'repeat' takes exactly one integer repetition count");
            goto fail;
        }
        if (in.targs[0].val == 0) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'repeat 0' is not allowed: the count must be at least 1");
            goto fail;
        }
        in.rep_count = in.targs[0].val;
    }

    if (kind != DI_REPEAT && p->s[p->pos] == '{') {
        dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                "unexpected '{' after '%s': only 'repeat' opens a block",
                name);
        goto fail;
    }

    if (kind == DI_REPEAT) {
        if (depth + 1 >= DEM_MAX_NEST) {
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "'repeat' blocks nested more than %d deep", DEM_MAX_NEST);
            goto fail;
        }
        dp_skip_all(p);
        if (p->s[p->pos] != '{') {
            dp_snip(p, p->pos, tok, sizeof tok);
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, line,
                    "expected '{' after 'repeat %llu', got '%s'",
                    (unsigned long long)in.rep_count, tok);
            goto fail;
        }
        p->pos++;
        in.body = (dem_block_t*)calloc(1, sizeof(dem_block_t));
        if (!in.body) {
            dp_fail(p, MOONLAB_STIM_ERR_OOM, line,
                    "out of memory opening a 'repeat' block");
            goto fail;
        }
        if (dp_block(p, in.body, depth + 1) != 0) goto fail;
    }

    {
        void* np = dem_grow(blk->v, &blk->cap, blk->n + 1, sizeof(dem_inst_t));
        if (!np) {
            dp_fail(p, MOONLAB_STIM_ERR_OOM, line,
                    "out of memory storing instruction '%s'", name);
            goto fail;
        }
        blk->v = (dem_inst_t*)np;
        blk->v[blk->n++] = in;
    }
    return 0;

fail:
    dem_inst_release(&in);
    return -1;
}

static int dp_block(dem_parser_t* p, dem_block_t* blk, int depth) {
    for (;;) {
        dp_skip_all(p);
        char c = p->s[p->pos];
        if (c == '\0') {
            if (depth > 0) {
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, p->line,
                        "unterminated 'repeat' block: expected '}' before the "
                        "end of the file");
                return -1;
            }
            return 0;
        }
        if (c == '}') {
            if (depth == 0) {
                dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, p->line,
                        "unexpected '}': no 'repeat' block is open");
                return -1;
            }
            p->pos++;
            return 0;
        }
        if (!dp_is_alpha(c)) {
            char tok[48];
            dp_snip(p, p->pos, tok, sizeof tok);
            dp_fail(p, MOONLAB_STIM_ERR_SYNTAX, p->line,
                    "expected an instruction name, got '%s'", tok);
            return -1;
        }
        if (dp_instruction(p, blk, depth) != 0) return -1;
    }
}

/* ================================================================== */
/*  Flattening: instruction tree -> edge list                          */
/* ================================================================== */

typedef struct { uint32_t a, b; uint64_t obs; double p; } dem_bedge_t;
typedef struct { uint32_t u, v; double q; } dem_bcorr_t;
typedef struct { uint32_t det; size_t off, n; } dem_bcoord_t;
/** One flattened mechanism, kept so moonlab_dem_to_text() can replay it. */
typedef struct { double p; size_t off, n; } dem_bmech_t;

/* Mechanism target encoding: kind in the high word, value in the low word. */
#define DEM_MT_DET 0u
#define DEM_MT_OBS 1u
#define DEM_MT_SEP 2u
#define DEM_MT_PACK(kind, val) (((uint64_t)(kind) << 32) | (uint64_t)(val))
#define DEM_MT_KIND(x) ((uint32_t)((x) >> 32))
#define DEM_MT_VAL(x)  ((uint32_t)((x) & 0xffffffffu))

typedef struct {
    dem_bedge_t*  edges; size_t nedge, ecap;
    size_t*       eslot; size_t eslots;   /* open addressing, stores index+1 */

    dem_bcorr_t*  corr;  size_t ncorr, ccap;
    size_t*       cslot; size_t cslots;

    dem_bmech_t*  mech;  size_t nmech, mcap;
    uint64_t*     mtarg; size_t nmtarg, mtcap;

    dem_bcoord_t* crec;  size_t ncrec, crcap;
    double*       cpool; size_t ncpool, cpcap;

    uint32_t*     mkeys; size_t nmkeys, mkcap; /* scratch, per mechanism */

    size_t ndet, nobs, nhyper, ndetless;
    int    oom;
} dem_build_t;

typedef struct {
    uint64_t det_offset;
    double*  cshift;
    size_t   ncshift, cscap;
} dem_state_t;

static uint64_t dem_mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

/* --- edge table ---------------------------------------------------- */

static int dem_edge_rehash(dem_build_t* B, size_t want) {
    size_t cap = B->eslots ? B->eslots : 64;
    while (cap < want * 2) {
        if (cap > SIZE_MAX / 2) return -1;
        cap *= 2;
    }
    size_t* ns = (size_t*)calloc(cap, sizeof(size_t));
    if (!ns) return -1;
    for (size_t i = 0; i < B->nedge; i++) {
        uint64_t h = dem_mix(dem_mix(dem_mix(0x1234567u, B->edges[i].a),
                                     B->edges[i].b), B->edges[i].obs);
        size_t j = (size_t)h & (cap - 1);
        while (ns[j]) j = (j + 1) & (cap - 1);
        ns[j] = i + 1;
    }
    free(B->eslot);
    B->eslot = ns;
    B->eslots = cap;
    return 0;
}

/** Find or insert the key and XOR-merge @p p into it.
 *  @return the edge index, or SIZE_MAX on allocation failure. */
static size_t dem_edge_merge(dem_build_t* B, uint32_t a, uint32_t b,
                             uint64_t obs, double p) {
    if (B->nedge * 2 + 2 > B->eslots)
        if (dem_edge_rehash(B, B->nedge + 1) != 0) return SIZE_MAX;
    uint64_t h = dem_mix(dem_mix(dem_mix(0x1234567u, a), b), obs);
    size_t j = (size_t)h & (B->eslots - 1);
    while (B->eslot[j]) {
        size_t idx = B->eslot[j] - 1;
        if (B->edges[idx].a == a && B->edges[idx].b == b &&
            B->edges[idx].obs == obs) {
            double q = B->edges[idx].p;
            B->edges[idx].p = p * (1.0 - q) + q * (1.0 - p);
            return idx;
        }
        j = (j + 1) & (B->eslots - 1);
    }
    void* np = dem_grow(B->edges, &B->ecap, B->nedge + 1, sizeof(dem_bedge_t));
    if (!np) return SIZE_MAX;
    B->edges = (dem_bedge_t*)np;
    size_t idx = B->nedge++;
    B->edges[idx].a = a;
    B->edges[idx].b = b;
    B->edges[idx].obs = obs;
    B->edges[idx].p = p; /* == p * (1 - 0) + 0 * (1 - p) */
    B->eslot[j] = idx + 1;
    return idx;
}

/* --- correlation table --------------------------------------------- */

static int dem_corr_rehash(dem_build_t* B, size_t want) {
    size_t cap = B->cslots ? B->cslots : 64;
    while (cap < want * 2) {
        if (cap > SIZE_MAX / 2) return -1;
        cap *= 2;
    }
    size_t* ns = (size_t*)calloc(cap, sizeof(size_t));
    if (!ns) return -1;
    for (size_t i = 0; i < B->ncorr; i++) {
        uint64_t h = dem_mix(dem_mix(0x89abcdefu, B->corr[i].u), B->corr[i].v);
        size_t j = (size_t)h & (cap - 1);
        while (ns[j]) j = (j + 1) & (cap - 1);
        ns[j] = i + 1;
    }
    free(B->cslot);
    B->cslot = ns;
    B->cslots = cap;
    return 0;
}

static int dem_corr_merge(dem_build_t* B, uint32_t u, uint32_t v, double p) {
    if (B->ncorr * 2 + 2 > B->cslots)
        if (dem_corr_rehash(B, B->ncorr + 1) != 0) return -1;
    uint64_t h = dem_mix(dem_mix(0x89abcdefu, u), v);
    size_t j = (size_t)h & (B->cslots - 1);
    while (B->cslot[j]) {
        size_t idx = B->cslot[j] - 1;
        if (B->corr[idx].u == u && B->corr[idx].v == v) {
            double q = B->corr[idx].q;
            B->corr[idx].q = p * (1.0 - q) + q * (1.0 - p);
            return 0;
        }
        j = (j + 1) & (B->cslots - 1);
    }
    void* np = dem_grow(B->corr, &B->ccap, B->ncorr + 1, sizeof(dem_bcorr_t));
    if (!np) return -1;
    B->corr = (dem_bcorr_t*)np;
    size_t idx = B->ncorr++;
    B->corr[idx].u = u;
    B->corr[idx].v = v;
    B->corr[idx].q = p;
    B->cslot[j] = idx + 1;
    return 0;
}

static int dem_cmp_u32(const void* x, const void* y) {
    uint32_t a = *(const uint32_t*)x, b = *(const uint32_t*)y;
    return (a > b) - (a < b);
}

static int dem_cmp_corr(const void* x, const void* y) {
    const dem_bcorr_t* a = (const dem_bcorr_t*)x;
    const dem_bcorr_t* b = (const dem_bcorr_t*)y;
    if (a->u != b->u) return a->u < b->u ? -1 : 1;
    if (a->v != b->v) return a->v < b->v ? -1 : 1;
    return 0;
}

static int dem_push_mtarg(dem_build_t* B, uint64_t t) {
    void* np = dem_grow(B->mtarg, &B->mtcap, B->nmtarg + 1, sizeof(uint64_t));
    if (!np) return -1;
    B->mtarg = (uint64_t*)np;
    B->mtarg[B->nmtarg++] = t;
    return 0;
}

static int dem_flatten_error(dem_build_t* B, const dem_inst_t* in,
                             dem_state_t* st, moonlab_stim_error_t* err) {
    /* Detector and observable counts follow every reference in the file,
     * including mechanisms whose probability makes them no-ops. */
    for (size_t i = 0; i < in->ntargs; i++) {
        if (in->targs[i].kind == DT_DET) {
            uint64_t abs_d = st->det_offset + in->targs[i].val;
            if (abs_d >= (uint64_t)MOONLAB_UF_BOUNDARY) {
                dem_err_set(err, MOONLAB_STIM_ERR_OVERFLOW, in->line,
                            "detector index %llu is out of range: %u is "
                            "reserved as the boundary sentinel",
                            (unsigned long long)abs_d,
                            (unsigned)MOONLAB_UF_BOUNDARY);
                return -1;
            }
            if (abs_d + 1 > B->ndet) B->ndet = (size_t)abs_d + 1;
        } else if (in->targs[i].kind == DT_OBS) {
            if (in->targs[i].val + 1 > B->nobs)
                B->nobs = (size_t)in->targs[i].val + 1;
        }
    }

    const double p = in->args[0];
    if (p <= 0.0 || p >= 1.0) return 0; /* never fires, or always fires */

    /* Record the mechanism verbatim, for faithful re-serialisation. */
    const size_t mech_off = B->nmtarg;
    for (size_t i = 0; i < in->ntargs; i++) {
        uint64_t packed;
        if (in->targs[i].kind == DT_DET)
            packed = DEM_MT_PACK(DEM_MT_DET,
                                 st->det_offset + in->targs[i].val);
        else if (in->targs[i].kind == DT_OBS)
            packed = DEM_MT_PACK(DEM_MT_OBS, in->targs[i].val);
        else
            packed = DEM_MT_PACK(DEM_MT_SEP, 0);
        if (dem_push_mtarg(B, packed) != 0) { B->oom = 1; return -1; }
    }
    {
        void* np = dem_grow(B->mech, &B->mcap, B->nmech + 1,
                            sizeof(dem_bmech_t));
        if (!np) { B->oom = 1; return -1; }
        B->mech = (dem_bmech_t*)np;
        B->mech[B->nmech].p = p;
        B->mech[B->nmech].off = mech_off;
        B->mech[B->nmech].n = B->nmtarg - mech_off;
        B->nmech++;
    }

    /* Split on '^'; every graphlike component becomes an edge. */
    B->nmkeys = 0;
    size_t comp_ndet = 0;
    uint32_t d0 = 0, d1 = 0;
    uint64_t obs = 0;
    size_t i = 0;
    for (;;) {
        int end = (i == in->ntargs);
        if (end || in->targs[i].kind == DT_SEP) {
            if (comp_ndet == 0) {
                B->ndetless++;
            } else if (comp_ndet > 2) {
                B->nhyper++;
            } else {
                uint32_t a = d0;
                uint32_t b = (comp_ndet == 2) ? d1 : MOONLAB_UF_BOUNDARY;
                if (a > b) { uint32_t t = a; a = b; b = t; }
                size_t idx = dem_edge_merge(B, a, b, obs, p);
                if (idx == SIZE_MAX) { B->oom = 1; return -1; }
                void* np = dem_grow(B->mkeys, &B->mkcap, B->nmkeys + 1,
                                    sizeof(uint32_t));
                if (!np) { B->oom = 1; return -1; }
                B->mkeys = (uint32_t*)np;
                B->mkeys[B->nmkeys++] = (uint32_t)idx;
            }
            comp_ndet = 0;
            obs = 0;
            if (end) break;
            i++;
            continue;
        }
        if (in->targs[i].kind == DT_DET) {
            uint32_t abs_d = (uint32_t)(st->det_offset + in->targs[i].val);
            if (comp_ndet == 0) d0 = abs_d;
            else if (comp_ndet == 1) d1 = abs_d;
            comp_ndet++;
        } else { /* DT_OBS */
            obs |= (uint64_t)1u << in->targs[i].val;
        }
        i++;
    }

    /* Every pair of components of one mechanism is a correlation link. */
    if (B->nmkeys >= 2) {
        qsort(B->mkeys, B->nmkeys, sizeof(uint32_t), dem_cmp_u32);
        size_t uniq = 0;
        for (size_t k = 0; k < B->nmkeys; k++)
            if (k == 0 || B->mkeys[k] != B->mkeys[k - 1])
                B->mkeys[uniq++] = B->mkeys[k];
        for (size_t x = 0; x < uniq; x++)
            for (size_t y = x + 1; y < uniq; y++)
                if (dem_corr_merge(B, B->mkeys[x], B->mkeys[y], p) != 0) {
                    B->oom = 1;
                    return -1;
                }
    }
    return 0;
}

static int dem_flatten_block(dem_build_t* B, const dem_block_t* blk,
                             dem_state_t* st, moonlab_stim_error_t* err,
                             int depth) {
    if (depth >= DEM_MAX_NEST) {
        dem_err_set(err, MOONLAB_STIM_ERR_SYNTAX, 0,
                    "'repeat' blocks nested more than %d deep", DEM_MAX_NEST);
        return -1;
    }
    for (size_t i = 0; i < blk->n; i++) {
        const dem_inst_t* in = &blk->v[i];
        switch (in->kind) {
        case DI_ERROR:
            if (dem_flatten_error(B, in, st, err) != 0) {
                if (B->oom)
                    dem_err_set(err, MOONLAB_STIM_ERR_OOM, in->line,
                                "out of memory building the edge list");
                return -1;
            }
            break;

        case DI_DETECTOR: {
            /* Coordinates carry the accumulated shift_detectors offset. */
            size_t off = B->ncpool;
            if (in->nargs > 0) {
                void* np = dem_grow(B->cpool, &B->cpcap,
                                    B->ncpool + in->nargs, sizeof(double));
                if (!np) {
                    dem_err_set(err, MOONLAB_STIM_ERR_OOM, in->line,
                                "out of memory storing detector coordinates");
                    return -1;
                }
                B->cpool = (double*)np;
                for (size_t k = 0; k < in->nargs; k++) {
                    double sh = (k < st->ncshift) ? st->cshift[k] : 0.0;
                    B->cpool[B->ncpool + k] = in->args[k] + sh;
                }
                B->ncpool += in->nargs;
            }
            for (size_t t = 0; t < in->ntargs; t++) {
                uint64_t abs_d = st->det_offset + in->targs[t].val;
                if (abs_d >= (uint64_t)MOONLAB_UF_BOUNDARY) {
                    dem_err_set(err, MOONLAB_STIM_ERR_OVERFLOW, in->line,
                                "detector index %llu is out of range: %u is "
                                "reserved as the boundary sentinel",
                                (unsigned long long)abs_d,
                                (unsigned)MOONLAB_UF_BOUNDARY);
                    return -1;
                }
                if (abs_d + 1 > B->ndet) B->ndet = (size_t)abs_d + 1;
                void* np = dem_grow(B->crec, &B->crcap, B->ncrec + 1,
                                    sizeof(dem_bcoord_t));
                if (!np) {
                    dem_err_set(err, MOONLAB_STIM_ERR_OOM, in->line,
                                "out of memory storing detector coordinates");
                    return -1;
                }
                B->crec = (dem_bcoord_t*)np;
                B->crec[B->ncrec].det = (uint32_t)abs_d;
                B->crec[B->ncrec].off = off;
                B->crec[B->ncrec].n = in->nargs;
                B->ncrec++;
            }
            break;
        }

        case DI_LOGICAL:
            for (size_t t = 0; t < in->ntargs; t++)
                if (in->targs[t].val + 1 > B->nobs)
                    B->nobs = (size_t)in->targs[t].val + 1;
            break;

        case DI_SHIFT: {
            uint64_t shift = in->targs[0].val;
            if (st->det_offset > (uint64_t)MOONLAB_UF_BOUNDARY - shift) {
                dem_err_set(err, MOONLAB_STIM_ERR_OVERFLOW, in->line,
                            "'shift_detectors %llu' pushes the detector "
                            "offset past the %u index limit",
                            (unsigned long long)shift,
                            (unsigned)MOONLAB_UF_BOUNDARY);
                return -1;
            }
            st->det_offset += shift;
            if (in->nargs > st->ncshift) {
                void* np = dem_grow(st->cshift, &st->cscap, in->nargs,
                                    sizeof(double));
                if (!np) {
                    dem_err_set(err, MOONLAB_STIM_ERR_OOM, in->line,
                                "out of memory accumulating a coordinate "
                                "shift");
                    return -1;
                }
                st->cshift = (double*)np;
                for (size_t k = st->ncshift; k < in->nargs; k++)
                    st->cshift[k] = 0.0;
                st->ncshift = in->nargs;
            }
            for (size_t k = 0; k < in->nargs; k++) st->cshift[k] += in->args[k];
            break;
        }

        case DI_REPEAT:
            /* The state is threaded through, so a shift_detectors inside the
             * body is re-applied on every iteration -- which is how Stim
             * lays out a multi-round circuit's detector indices. */
            for (uint64_t r = 0; r < in->rep_count; r++)
                if (dem_flatten_block(B, in->body, st, err, depth + 1) != 0)
                    return -1;
            break;

        default:
            break;
        }
    }
    return 0;
}

/* ================================================================== */
/*  Model                                                              */
/* ================================================================== */

struct moonlab_dem {
    size_t       ndet, nobs;
    size_t       nedge;
    uint32_t*    ea;
    uint32_t*    eb;
    uint64_t*    eo;
    double*      ep;
    double*      ew;
    size_t       ncorr;
    uint32_t*    ca;
    uint32_t*    cb;
    double*      cq;
    size_t       nhyper;
    size_t       ndetless;
    size_t*      coord_off; /* ndet + 1 entries, NULL when none declared */
    double*      coord_val;
    size_t       nmech;
    dem_bmech_t* mech;
    uint64_t*    mtarg;
    size_t       nmtarg;
};

static void dem_build_release(dem_build_t* B) {
    free(B->edges);
    free(B->eslot);
    free(B->corr);
    free(B->cslot);
    free(B->mech);
    free(B->mtarg);
    free(B->crec);
    free(B->cpool);
    free(B->mkeys);
    memset(B, 0, sizeof *B);
}

void moonlab_dem_free(moonlab_dem_t* d) {
    if (!d) return;
    free(d->ea);
    free(d->eb);
    free(d->eo);
    free(d->ep);
    free(d->ew);
    free(d->ca);
    free(d->cb);
    free(d->cq);
    free(d->coord_off);
    free(d->coord_val);
    free(d->mech);
    free(d->mtarg);
    free(d);
}

static moonlab_dem_t* dem_finish(dem_build_t* B, moonlab_stim_error_t* err) {
    moonlab_dem_t* d = (moonlab_dem_t*)calloc(1, sizeof(*d));
    if (!d) {
        dem_err_set(err, MOONLAB_STIM_ERR_OOM, 0,
                    "out of memory allocating the detector error model");
        return NULL;
    }
    d->ndet     = B->ndet;
    d->nobs     = B->nobs;
    d->nedge    = B->nedge;
    d->ncorr    = B->ncorr;
    d->nhyper   = B->nhyper;
    d->ndetless = B->ndetless;

    if (B->nedge > 0) {
        d->ea = (uint32_t*)malloc(B->nedge * sizeof(uint32_t));
        d->eb = (uint32_t*)malloc(B->nedge * sizeof(uint32_t));
        d->eo = (uint64_t*)malloc(B->nedge * sizeof(uint64_t));
        d->ep = (double*)malloc(B->nedge * sizeof(double));
        d->ew = (double*)malloc(B->nedge * sizeof(double));
        if (!d->ea || !d->eb || !d->eo || !d->ep || !d->ew) goto oom;
        for (size_t i = 0; i < B->nedge; i++) {
            d->ea[i] = B->edges[i].a;
            d->eb[i] = B->edges[i].b;
            d->eo[i] = B->edges[i].obs;
            d->ep[i] = B->edges[i].p;
            d->ew[i] = log((1.0 - B->edges[i].p) / B->edges[i].p);
        }
    }

    if (B->ncorr > 0) {
        qsort(B->corr, B->ncorr, sizeof(dem_bcorr_t), dem_cmp_corr);
        d->ca = (uint32_t*)malloc(B->ncorr * sizeof(uint32_t));
        d->cb = (uint32_t*)malloc(B->ncorr * sizeof(uint32_t));
        d->cq = (double*)malloc(B->ncorr * sizeof(double));
        if (!d->ca || !d->cb || !d->cq) goto oom;
        for (size_t i = 0; i < B->ncorr; i++) {
            d->ca[i] = B->corr[i].u;
            d->cb[i] = B->corr[i].v;
            d->cq[i] = B->corr[i].q;
        }
    }

    if (B->ncrec > 0 && B->ndet > 0) {
        /* Last declaration wins, as it does for a re-declared detector in
         * Stim. */
        size_t* off = (size_t*)calloc(B->ndet + 1, sizeof(size_t));
        size_t* pick = (size_t*)malloc(B->ndet * sizeof(size_t));
        if (!off || !pick) { free(off); free(pick); goto oom; }
        for (size_t i = 0; i < B->ndet; i++) pick[i] = SIZE_MAX;
        for (size_t i = 0; i < B->ncrec; i++) pick[B->crec[i].det] = i;
        size_t total = 0;
        for (size_t i = 0; i < B->ndet; i++) {
            off[i] = total;
            if (pick[i] != SIZE_MAX) total += B->crec[pick[i]].n;
        }
        off[B->ndet] = total;
        double* vals = NULL;
        if (total > 0) {
            vals = (double*)malloc(total * sizeof(double));
            if (!vals) { free(off); free(pick); goto oom; }
            for (size_t i = 0; i < B->ndet; i++) {
                if (pick[i] == SIZE_MAX) continue;
                const dem_bcoord_t* r = &B->crec[pick[i]];
                if (r->n) memcpy(vals + off[i], B->cpool + r->off,
                                 r->n * sizeof(double));
            }
        }
        free(pick);
        d->coord_off = off;
        d->coord_val = vals;
    }

    /* The mechanism list is handed over wholesale; it is what to_text()
     * replays. */
    d->nmech  = B->nmech;
    d->mech   = B->mech;
    d->mtarg  = B->mtarg;
    d->nmtarg = B->nmtarg;
    B->mech   = NULL;
    B->mtarg  = NULL;
    B->nmech  = 0;
    B->nmtarg = 0;
    return d;

oom:
    dem_err_set(err, MOONLAB_STIM_ERR_OOM, 0,
                "out of memory allocating the detector error model");
    moonlab_dem_free(d);
    return NULL;
}

moonlab_dem_t* moonlab_dem_parse(const char* text, moonlab_stim_error_t* err) {
    dem_err_clear(err);
    if (!text) {
        dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                    "moonlab_dem_parse: text is NULL");
        return NULL;
    }

    dem_parser_t p;
    memset(&p, 0, sizeof p);
    p.s = text;
    p.line = 1;
    p.err = err;

    dem_block_t root;
    memset(&root, 0, sizeof root);
    if (dp_block(&p, &root, 0) != 0) {
        dem_block_free(&root);
        if (err && err->code == MOONLAB_STIM_OK)
            dem_err_set(err, MOONLAB_STIM_ERR_SYNTAX, p.line,
                        "detector error model parse failed");
        return NULL;
    }

    dem_build_t B;
    memset(&B, 0, sizeof B);
    dem_state_t st;
    memset(&st, 0, sizeof st);

    moonlab_dem_t* d = NULL;
    if (dem_flatten_block(&B, &root, &st, err, 0) == 0)
        d = dem_finish(&B, err);

    free(st.cshift);
    dem_build_release(&B);
    dem_block_free(&root);
    return d;
}

moonlab_dem_t* moonlab_dem_parse_file(const char* path,
                                      moonlab_stim_error_t* err) {
    dem_err_clear(err);
    if (!path) {
        dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                    "moonlab_dem_parse_file: path is NULL");
        return NULL;
    }
    FILE* f = fopen(path, "rb");
    if (!f) {
        dem_err_set(err, MOONLAB_STIM_ERR_IO, 0,
                    "cannot open detector error model file '%s'", path);
        return NULL;
    }
    dem_sb_t b;
    memset(&b, 0, sizeof b);
    char chunk[8192];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (sb_reserve(&b, got) != 0) {
            free(b.s);
            fclose(f);
            dem_err_set(err, MOONLAB_STIM_ERR_OOM, 0,
                        "out of memory reading '%s'", path);
            return NULL;
        }
        memcpy(b.s + b.len, chunk, got);
        b.len += got;
        b.s[b.len] = '\0';
    }
    int bad = ferror(f);
    fclose(f);
    if (bad) {
        free(b.s);
        dem_err_set(err, MOONLAB_STIM_ERR_IO, 0, "error reading '%s'", path);
        return NULL;
    }
    if (!b.s) {
        if (sb_reserve(&b, 0) != 0) {
            dem_err_set(err, MOONLAB_STIM_ERR_OOM, 0,
                        "out of memory reading '%s'", path);
            return NULL;
        }
        b.s[0] = '\0';
    }
    moonlab_dem_t* d = moonlab_dem_parse(b.s, err);
    free(b.s);
    return d;
}

/* ================================================================== */
/*  Accessors                                                          */
/* ================================================================== */

size_t moonlab_dem_num_detectors(const moonlab_dem_t* d) {
    return d ? d->ndet : 0;
}
size_t moonlab_dem_num_observables(const moonlab_dem_t* d) {
    return d ? d->nobs : 0;
}
size_t moonlab_dem_num_edges(const moonlab_dem_t* d) {
    return d ? d->nedge : 0;
}
size_t moonlab_dem_num_correlations(const moonlab_dem_t* d) {
    return d ? d->ncorr : 0;
}
size_t moonlab_dem_num_hyperedges(const moonlab_dem_t* d) {
    return d ? d->nhyper : 0;
}
size_t moonlab_dem_num_detectorless(const moonlab_dem_t* d) {
    return d ? d->ndetless : 0;
}

long moonlab_dem_edges(const moonlab_dem_t* d,
                       uint32_t* edge_a, uint32_t* edge_b,
                       double* edge_weight, uint64_t* edge_obs,
                       double* edge_prob, size_t cap) {
    if (!d) return MOONLAB_STIM_ERR_BAD_ARG;
    if (cap < d->nedge) return MOONLAB_STIM_ERR_OVERFLOW;
    if (d->nedge > 0) {
        if (edge_a)      memcpy(edge_a, d->ea, d->nedge * sizeof(uint32_t));
        if (edge_b)      memcpy(edge_b, d->eb, d->nedge * sizeof(uint32_t));
        if (edge_weight) memcpy(edge_weight, d->ew, d->nedge * sizeof(double));
        if (edge_obs)    memcpy(edge_obs, d->eo, d->nedge * sizeof(uint64_t));
        if (edge_prob)   memcpy(edge_prob, d->ep, d->nedge * sizeof(double));
    }
    return (long)d->nedge;
}

long moonlab_dem_correlations(const moonlab_dem_t* d,
                              uint32_t* corr_a, uint32_t* corr_b,
                              double* corr_joint_p, size_t cap) {
    if (!d) return MOONLAB_STIM_ERR_BAD_ARG;
    if (cap < d->ncorr) return MOONLAB_STIM_ERR_OVERFLOW;
    if (d->ncorr > 0) {
        if (corr_a)       memcpy(corr_a, d->ca, d->ncorr * sizeof(uint32_t));
        if (corr_b)       memcpy(corr_b, d->cb, d->ncorr * sizeof(uint32_t));
        if (corr_joint_p) memcpy(corr_joint_p, d->cq,
                                 d->ncorr * sizeof(double));
    }
    return (long)d->ncorr;
}

long moonlab_dem_detector_coords(const moonlab_dem_t* d, size_t detector,
                                 double* out, size_t cap) {
    if (!d || detector >= d->ndet) return -1;
    if (!d->coord_off) return 0;
    size_t off = d->coord_off[detector];
    size_t n = d->coord_off[detector + 1] - off;
    if (out && n > 0) {
        size_t k = n < cap ? n : cap;
        if (k > 0) memcpy(out, d->coord_val + off, k * sizeof(double));
    }
    return (long)n;
}

moonlab_uf_decoder_t* moonlab_dem_make_uf_decoder(const moonlab_dem_t* d,
                                                  int correlated) {
    if (!d) return NULL;
    if (!correlated || d->ncorr == 0)
        return moonlab_uf_decoder_new(d->ndet, d->nobs, d->ea, d->eb,
                                      d->ew, d->eo, d->nedge);
    /* The correlated constructor validates the links itself (0 < q < 0.5,
     * 0 < p < 1) and refuses the model when one is out of range.  The links
     * are handed over untouched: quietly dropping the offending one would
     * decode a different model than the file describes. */
    return moonlab_uf_decoder_new_correlated(d->ndet, d->nobs, d->ea, d->eb,
                                             d->ew, d->eo, d->nedge, d->ep,
                                             d->ca, d->cb, d->cq, d->ncorr);
}

/* ================================================================== */
/*  Serialisation                                                      */
/* ================================================================== */

/** `detector(...) D#` lines, a bare `detector D#` pinning the detector count
 *  when the top index is otherwise unreferenced, and `logical_observable`
 *  declarations for observables no mechanism flips. */
static int dem_emit_declarations(dem_sb_t* b, size_t ndet, size_t nobs,
                                 const size_t* coord_off,
                                 const double* coord_val,
                                 uint64_t obs_touched, size_t det_max_ref) {
    if (coord_off) {
        for (size_t i = 0; i < ndet; i++) {
            size_t n = coord_off[i + 1] - coord_off[i];
            if (n == 0) continue;
            if (sb_add(b, "detector(") != 0) return -1;
            for (size_t k = 0; k < n; k++) {
                if (k && sb_add(b, ", ") != 0) return -1;
                if (sb_addf(b, "%.17g", coord_val[coord_off[i] + k]) != 0)
                    return -1;
            }
            if (sb_addf(b, ") D%zu\n", i) != 0) return -1;
            if (i + 1 > det_max_ref) det_max_ref = i + 1;
        }
    }
    if (ndet > det_max_ref)
        if (sb_addf(b, "detector D%zu\n", ndet - 1) != 0) return -1;
    for (size_t o = 0; o < nobs && o < 64; o++)
        if (((obs_touched >> o) & 1u) == 0)
            if (sb_addf(b, "logical_observable L%zu\n", o) != 0) return -1;
    return 0;
}

static int dem_emit_edge_targets(dem_sb_t* b, uint32_t a, uint32_t bb,
                                 uint64_t obs) {
    if (sb_addf(b, " D%u", a) != 0) return -1;
    if (bb != MOONLAB_UF_BOUNDARY)
        if (sb_addf(b, " D%u", bb) != 0) return -1;
    for (unsigned i = 0; i < 64; i++)
        if ((obs >> i) & 1u)
            if (sb_addf(b, " L%u", i) != 0) return -1;
    return 0;
}

char* moonlab_dem_to_text(const moonlab_dem_t* d) {
    if (!d) return NULL;
    dem_sb_t b;
    memset(&b, 0, sizeof b);

    uint64_t obs_touched = 0;
    size_t det_max_ref = 0;
    for (size_t i = 0; i < d->nmtarg; i++) {
        uint32_t k = DEM_MT_KIND(d->mtarg[i]);
        uint32_t v = DEM_MT_VAL(d->mtarg[i]);
        if (k == DEM_MT_DET) {
            if ((size_t)v + 1 > det_max_ref) det_max_ref = (size_t)v + 1;
        } else if (k == DEM_MT_OBS) {
            obs_touched |= (uint64_t)1u << v;
        }
    }
    if (dem_emit_declarations(&b, d->ndet, d->nobs, d->coord_off,
                              d->coord_val, obs_touched, det_max_ref) != 0)
        goto oom;

    /* Replay the flattened mechanisms exactly as they were read, so
     * non-graphlike and observable-only mechanisms survive the round trip
     * alongside the graph edges. */
    for (size_t m = 0; m < d->nmech; m++) {
        if (sb_addf(&b, "error(%.17g)", d->mech[m].p) != 0) goto oom;
        /* A mechanism can legitimately carry no targets at all -- an
         * `error(p)` with an empty target list -- so index the pool rather
         * than forming a pointer into it. */
        for (size_t i = 0; i < d->mech[m].n; i++) {
            uint64_t packed = d->mtarg[d->mech[m].off + i];
            uint32_t k = DEM_MT_KIND(packed);
            uint32_t v = DEM_MT_VAL(packed);
            int rc;
            if (k == DEM_MT_DET)      rc = sb_addf(&b, " D%u", v);
            else if (k == DEM_MT_OBS) rc = sb_addf(&b, " L%u", v);
            else                      rc = sb_add(&b, " ^");
            if (rc != 0) goto oom;
        }
        if (sb_add(&b, "\n") != 0) goto oom;
    }
    if (!b.s) {
        b.s = (char*)calloc(1, 1);
        if (!b.s) return NULL;
    }
    return b.s;

oom:
    free(b.s);
    return NULL;
}

char* moonlab_dem_text_from_edges(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_prob, const uint64_t* edge_obs, size_t num_edges,
    const uint32_t* corr_a, const uint32_t* corr_b,
    const double* corr_joint_p, size_t num_corr,
    moonlab_stim_error_t* err) {
    dem_err_clear(err);

    if (num_edges > 0 && (!edge_a || !edge_b || !edge_prob || !edge_obs)) {
        dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                    "moonlab_dem_text_from_edges: edge arrays are NULL but "
                    "num_edges is %zu", num_edges);
        return NULL;
    }
    if (num_corr > 0 && (!corr_a || !corr_b || !corr_joint_p)) {
        dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                    "moonlab_dem_text_from_edges: correlation arrays are NULL "
                    "but num_corr is %zu", num_corr);
        return NULL;
    }
    if (num_observables > 64) {
        dem_err_set(err, MOONLAB_STIM_ERR_UNSUPPORTED, 0,
                    "%zu observables exceeds the 64 the decoder's uint64_t "
                    "observable mask can carry", num_observables);
        return NULL;
    }

    uint64_t obs_touched = 0;
    size_t det_max_ref = 0;
    for (size_t e = 0; e < num_edges; e++) {
        if (edge_a[e] == MOONLAB_UF_BOUNDARY) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "edge %zu has no detector endpoint: edge_a is the "
                        "boundary sentinel", e);
            return NULL;
        }
        if (edge_b[e] != MOONLAB_UF_BOUNDARY && edge_a[e] > edge_b[e]) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "edge %zu has endpoints out of order (D%u > D%u); "
                        "moonlab_dem_edges reports a <= b and the round trip "
                        "needs the same convention",
                        e, edge_a[e], edge_b[e]);
            return NULL;
        }
        if ((size_t)edge_a[e] >= num_detectors ||
            (edge_b[e] != MOONLAB_UF_BOUNDARY &&
             (size_t)edge_b[e] >= num_detectors)) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "edge %zu references a detector at or beyond "
                        "num_detectors=%zu", e, num_detectors);
            return NULL;
        }
        if (!(edge_prob[e] > 0.0) || edge_prob[e] >= 1.0) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "edge %zu has probability %.17g, outside the open "
                        "interval (0, 1)", e, edge_prob[e]);
            return NULL;
        }
        if (num_observables < 64 && (edge_obs[e] >> num_observables) != 0) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "edge %zu flips an observable at or beyond "
                        "num_observables=%zu", e, num_observables);
            return NULL;
        }
        obs_touched |= edge_obs[e];
        size_t hi = (size_t)edge_a[e] + 1;
        if (edge_b[e] != MOONLAB_UF_BOUNDARY && (size_t)edge_b[e] + 1 > hi)
            hi = (size_t)edge_b[e] + 1;
        if (hi > det_max_ref) det_max_ref = hi;
    }

    size_t*  link_off   = NULL;
    size_t*  link_idx   = NULL;
    double*  resid      = NULL;
    uint8_t* seen       = NULL;
    uint8_t* link_done  = NULL;
    uint8_t* resid_done = NULL;
    uint8_t* emit_resid = NULL;
    dem_sb_t b;
    memset(&b, 0, sizeof b);

    link_off = (size_t*)calloc(num_edges + 1, sizeof(size_t));
    if (!link_off) goto oom;

    /* Links must be in range, inside the (0, 0.5) window the peel inverse
     * needs, and free of duplicate pairs: a repeated pair would come back
     * from a re-import merged into one link with a different joint
     * probability, so the round trip would not close. */
    for (size_t k = 0; k < num_corr; k++) {
        if ((size_t)corr_a[k] >= num_edges || (size_t)corr_b[k] >= num_edges) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "correlation link %zu references edge index %u/%u "
                        "beyond num_edges=%zu",
                        k, corr_a[k], corr_b[k], num_edges);
            goto fail;
        }
        if (corr_a[k] == corr_b[k]) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "correlation link %zu joins edge %u to itself",
                        k, corr_a[k]);
            goto fail;
        }
        if (!(corr_joint_p[k] > 0.0) || corr_joint_p[k] >= 0.5) {
            dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                        "correlation link %zu has joint probability %.17g, "
                        "outside the open interval (0, 0.5)",
                        k, corr_joint_p[k]);
            goto fail;
        }
        link_off[corr_a[k] + 1]++;
        link_off[corr_b[k] + 1]++;
    }
    for (size_t e = 0; e < num_edges; e++) link_off[e + 1] += link_off[e];
    if (num_corr > 0) {
        link_idx = (size_t*)malloc(2 * num_corr * sizeof(size_t));
        if (!link_idx) goto oom;
        size_t* cur = (size_t*)malloc((num_edges + 1) * sizeof(size_t));
        if (!cur) goto oom;
        memcpy(cur, link_off, (num_edges + 1) * sizeof(size_t));
        for (size_t k = 0; k < num_corr; k++) {
            link_idx[cur[corr_a[k]]++] = k;
            link_idx[cur[corr_b[k]]++] = k;
        }
        free(cur);
        for (size_t e = 0; e < num_edges; e++) {
            for (size_t i = link_off[e]; i < link_off[e + 1]; i++) {
                size_t ka = link_idx[i];
                uint32_t oa = (corr_a[ka] == e) ? corr_b[ka] : corr_a[ka];
                for (size_t j = i + 1; j < link_off[e + 1]; j++) {
                    size_t kb = link_idx[j];
                    uint32_t ob = (corr_a[kb] == e) ? corr_b[kb] : corr_a[kb];
                    if (oa == ob) {
                        dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                                    "correlation links %zu and %zu are the "
                                    "same edge pair (%zu, %u); pre-combine "
                                    "them with q = q1(1-q2) + q2(1-q1)",
                                    ka, kb, e, oa);
                        goto fail;
                    }
                }
            }
        }
    }

    if (num_edges > 0) {
        resid      = (double*)malloc(num_edges * sizeof(double));
        emit_resid = (uint8_t*)calloc(num_edges, 1);
        seen       = (uint8_t*)calloc(num_edges, 1);
        resid_done = (uint8_t*)calloc(num_edges, 1);
        if (!resid || !emit_resid || !seen || !resid_done) goto oom;
    }
    if (num_corr > 0) {
        link_done = (uint8_t*)calloc(num_corr, 1);
        if (!link_done) goto oom;
    }

    /* Peel every joint probability touching an edge off that edge's total. */
    for (size_t e = 0; e < num_edges; e++) {
        double r = edge_prob[e];
        for (size_t i = link_off[e]; i < link_off[e + 1]; i++) {
            size_t k = link_idx[i];
            double q = corr_joint_p[k];
            r = (r - q) / (1.0 - 2.0 * q);
            if (r < -DEM_RESID_TOL) {
                dem_err_set(err, MOONLAB_STIM_ERR_BAD_ARG, 0,
                            "edge %zu has probability %.17g, too small to "
                            "carry its correlation links: peeling link %zu "
                            "(q = %.17g) leaves a residual of %.17g, so the "
                            "edge and joint probabilities are inconsistent",
                            e, edge_prob[e], k, q, r);
                goto fail;
            }
        }
        if (r < 0.0) r = 0.0;
        resid[e] = r;
        /* An edge with no links has to be emitted however small its
         * probability, or it would vanish from the model. */
        emit_resid[e] = (uint8_t)(r > DEM_RESID_SUPPRESS ||
                                  link_off[e + 1] == link_off[e]);
    }

    if (dem_emit_declarations(&b, num_detectors, num_observables, NULL, NULL,
                              obs_touched, det_max_ref) != 0)
        goto oom;

/* Emitting an instruction is what "introduces" the edges it mentions, and a
 * re-import indexes edges by first appearance -- so the emission order below
 * is what makes the round trip index-stable. */
#define DEM_EMIT_LINK(k) do {                                                 \
    size_t _k = (k);                                                          \
    uint32_t _u = corr_a[_k], _v = corr_b[_k];                                \
    if (_u > _v) { uint32_t _t = _u; _u = _v; _v = _t; }                      \
    if (sb_addf(&b, "error(%.17g)", corr_joint_p[_k]) != 0) goto oom;         \
    if (dem_emit_edge_targets(&b, edge_a[_u], edge_b[_u], edge_obs[_u]) != 0) \
        goto oom;                                                             \
    if (sb_add(&b, " ^") != 0) goto oom;                                      \
    if (dem_emit_edge_targets(&b, edge_a[_v], edge_b[_v], edge_obs[_v]) != 0) \
        goto oom;                                                             \
    if (sb_add(&b, "\n") != 0) goto oom;                                      \
    seen[_u] = 1;                                                             \
    seen[_v] = 1;                                                             \
    link_done[_k] = 1;                                                        \
} while (0)

#define DEM_EMIT_RESID(e) do {                                                \
    size_t _e = (e);                                                          \
    if (sb_addf(&b, "error(%.17g)", resid[_e]) != 0) goto oom;                \
    if (dem_emit_edge_targets(&b, edge_a[_e], edge_b[_e], edge_obs[_e]) != 0) \
        goto oom;                                                             \
    if (sb_add(&b, "\n") != 0) goto oom;                                      \
    seen[_e] = 1;                                                             \
    resid_done[_e] = 1;                                                       \
} while (0)

    for (size_t e = 0; e < num_edges; e++) {
        if (!seen[e]) {
            /* Prefer a link to an already introduced partner: that instruction
             * introduces this edge alone, keeping the index order intact. */
            size_t pick = SIZE_MAX;
            for (size_t i = link_off[e]; i < link_off[e + 1]; i++) {
                size_t k = link_idx[i];
                uint32_t other = (corr_a[k] == e) ? corr_b[k] : corr_a[k];
                if (!link_done[k] && seen[other]) { pick = k; break; }
            }
            if (pick != SIZE_MAX) {
                DEM_EMIT_LINK(pick);
            } else if (emit_resid[e]) {
                DEM_EMIT_RESID(e);
            } else {
                /* Fully explained by links, all of them to edges not yet
                 * introduced: take the lowest partner, which for a Stim
                 * decomposition is the very next edge index. */
                size_t best = SIZE_MAX;
                uint32_t best_other = 0;
                for (size_t i = link_off[e]; i < link_off[e + 1]; i++) {
                    size_t k = link_idx[i];
                    if (link_done[k]) continue;
                    uint32_t other = (corr_a[k] == e) ? corr_b[k] : corr_a[k];
                    if (best == SIZE_MAX || other < best_other) {
                        best = k;
                        best_other = other;
                    }
                }
                if (best != SIZE_MAX) DEM_EMIT_LINK(best);
            }
        }
        for (size_t i = link_off[e]; i < link_off[e + 1]; i++) {
            size_t k = link_idx[i];
            if (link_done[k]) continue;
            uint32_t hi = corr_a[k] > corr_b[k] ? corr_a[k] : corr_b[k];
            if ((size_t)hi == e) DEM_EMIT_LINK(k);
        }
        if (emit_resid[e] && !resid_done[e]) DEM_EMIT_RESID(e);
    }
    for (size_t k = 0; k < num_corr; k++)
        if (!link_done[k]) DEM_EMIT_LINK(k);

#undef DEM_EMIT_LINK
#undef DEM_EMIT_RESID

    if (!b.s) {
        b.s = (char*)calloc(1, 1);
        if (!b.s) goto oom;
    }
    free(link_off);
    free(link_idx);
    free(resid);
    free(seen);
    free(link_done);
    free(resid_done);
    free(emit_resid);
    return b.s;

oom:
    dem_err_set(err, MOONLAB_STIM_ERR_OOM, 0,
                "out of memory serialising the detector error model");
fail:
    free(b.s);
    free(link_off);
    free(link_idx);
    free(resid);
    free(seen);
    free(link_done);
    free(resid_done);
    free(emit_resid);
    return NULL;
}
