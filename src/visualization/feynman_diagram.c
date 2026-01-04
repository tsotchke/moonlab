/**
 * @file feynman_diagram.c
 * @brief Feynman Diagram Renderer Implementation
 */

#include "feynman_diagram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* ============================================================================
 * INTERNAL CONSTANTS
 * ============================================================================ */

#define PI 3.14159265358979323846

/* ============================================================================
 * STRING BUILDER (shared with circuit_diagram.c)
 * ============================================================================ */

typedef struct {
    char *buffer;
    size_t size;
    size_t capacity;
} string_builder_t;

static string_builder_t *sb_create(void) {
    string_builder_t *sb = malloc(sizeof(string_builder_t));
    if (!sb) return NULL;
    sb->capacity = 4096;
    sb->buffer = malloc(sb->capacity);
    if (!sb->buffer) {
        free(sb);
        return NULL;
    }
    sb->buffer[0] = '\0';
    sb->size = 0;
    return sb;
}

static void sb_free(string_builder_t *sb) {
    if (sb) {
        free(sb->buffer);
        free(sb);
    }
}

static void sb_ensure(string_builder_t *sb, size_t additional) {
    if (sb->size + additional >= sb->capacity) {
        size_t new_capacity = sb->capacity * 2;
        while (new_capacity < sb->size + additional) {
            new_capacity *= 2;
        }
        char *new_buffer = realloc(sb->buffer, new_capacity);
        if (new_buffer) {
            sb->buffer = new_buffer;
            sb->capacity = new_capacity;
        }
    }
}

static void sb_append(string_builder_t *sb, const char *str) {
    size_t len = strlen(str);
    sb_ensure(sb, len + 1);
    strcpy(sb->buffer + sb->size, str);
    sb->size += len;
}

static void sb_appendf(string_builder_t *sb, const char *fmt, ...) {
    char temp[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(temp, sizeof(temp), fmt, args);
    va_end(args);
    sb_append(sb, temp);
}

/* ============================================================================
 * DIAGRAM CREATION AND MANAGEMENT
 * ============================================================================ */

feynman_diagram_t *feynman_create(const char *process) {
    feynman_diagram_t *diagram = calloc(1, sizeof(feynman_diagram_t));
    if (!diagram) return NULL;

    diagram->num_vertices = 0;
    diagram->num_propagators = 0;
    diagram->loop_order = 0;
    diagram->min_x = 0;
    diagram->max_x = 0;
    diagram->min_y = 0;
    diagram->max_y = 0;

    if (process) {
        strncpy(diagram->process, process, FEYNMAN_MAX_TITLE_LEN - 1);
        diagram->process[FEYNMAN_MAX_TITLE_LEN - 1] = '\0';
    }

    return diagram;
}

void feynman_free(feynman_diagram_t *diagram) {
    free(diagram);
}

void feynman_set_title(feynman_diagram_t *diagram, const char *title) {
    if (diagram && title) {
        strncpy(diagram->title, title, FEYNMAN_MAX_TITLE_LEN - 1);
        diagram->title[FEYNMAN_MAX_TITLE_LEN - 1] = '\0';
    }
}

void feynman_set_loop_order(feynman_diagram_t *diagram, int order) {
    if (diagram) {
        diagram->loop_order = order;
    }
}

/* ============================================================================
 * ADDING VERTICES
 * ============================================================================ */

int feynman_add_vertex(feynman_diagram_t *diagram, double x, double y) {
    return feynman_add_vertex_labeled(diagram, x, y, "");
}

int feynman_add_vertex_labeled(feynman_diagram_t *diagram, double x, double y,
                               const char *label) {
    if (!diagram || diagram->num_vertices >= FEYNMAN_MAX_VERTICES) {
        return -1;
    }

    int id = diagram->num_vertices;
    feynman_vertex_t *v = &diagram->vertices[id];

    v->x = x;
    v->y = y;
    v->id = id;
    v->is_external = false;

    if (label) {
        strncpy(v->label, label, FEYNMAN_MAX_LABEL_LEN - 1);
        v->label[FEYNMAN_MAX_LABEL_LEN - 1] = '\0';
    } else {
        v->label[0] = '\0';
    }

    diagram->num_vertices++;
    feynman_update_bounds(diagram);
    return id;
}

int feynman_add_external_vertex(feynman_diagram_t *diagram, double x, double y,
                                const char *label) {
    int id = feynman_add_vertex_labeled(diagram, x, y, label);
    if (id >= 0) {
        diagram->vertices[id].is_external = true;
    }
    return id;
}

/* ============================================================================
 * ADDING PROPAGATORS
 * ============================================================================ */

static int add_propagator_internal(feynman_diagram_t *diagram, int from, int to,
                                   particle_type_t type, const char *label,
                                   bool arrow_forward) {
    if (!diagram || diagram->num_propagators >= FEYNMAN_MAX_PROPAGATORS) {
        return -1;
    }
    if (from < 0 || from >= diagram->num_vertices ||
        to < 0 || to >= diagram->num_vertices) {
        return -1;
    }

    feynman_propagator_t *p = &diagram->propagators[diagram->num_propagators];

    p->from_vertex = from;
    p->to_vertex = to;
    p->type = type;
    p->arrow_forward = arrow_forward;
    p->is_external = false;
    p->show_momentum = false;
    memset(p->momentum, 0, sizeof(p->momentum));

    if (label) {
        strncpy(p->label, label, FEYNMAN_MAX_LABEL_LEN - 1);
        p->label[FEYNMAN_MAX_LABEL_LEN - 1] = '\0';
    } else {
        p->label[0] = '\0';
    }

    diagram->num_propagators++;
    return 0;
}

int feynman_add_fermion(feynman_diagram_t *diagram, int from, int to, const char *label) {
    return add_propagator_internal(diagram, from, to, PARTICLE_FERMION, label, true);
}

int feynman_add_antifermion(feynman_diagram_t *diagram, int from, int to, const char *label) {
    return add_propagator_internal(diagram, from, to, PARTICLE_ANTIFERMION, label, false);
}

int feynman_add_photon(feynman_diagram_t *diagram, int from, int to, const char *label) {
    const char *lbl = (label && label[0]) ? label : "gamma";
    return add_propagator_internal(diagram, from, to, PARTICLE_PHOTON, lbl, true);
}

int feynman_add_gluon(feynman_diagram_t *diagram, int from, int to, const char *label) {
    const char *lbl = (label && label[0]) ? label : "g";
    return add_propagator_internal(diagram, from, to, PARTICLE_GLUON, lbl, true);
}

int feynman_add_w_boson(feynman_diagram_t *diagram, int from, int to, const char *label) {
    const char *lbl = (label && label[0]) ? label : "W";
    return add_propagator_internal(diagram, from, to, PARTICLE_W_BOSON, lbl, true);
}

int feynman_add_z_boson(feynman_diagram_t *diagram, int from, int to, const char *label) {
    const char *lbl = (label && label[0]) ? label : "Z";
    return add_propagator_internal(diagram, from, to, PARTICLE_Z_BOSON, lbl, true);
}

int feynman_add_higgs(feynman_diagram_t *diagram, int from, int to, const char *label) {
    const char *lbl = (label && label[0]) ? label : "H";
    return add_propagator_internal(diagram, from, to, PARTICLE_HIGGS, lbl, true);
}

int feynman_add_scalar(feynman_diagram_t *diagram, int from, int to, const char *label) {
    return add_propagator_internal(diagram, from, to, PARTICLE_SCALAR, label, true);
}

int feynman_add_propagator(feynman_diagram_t *diagram, int from, int to,
                           particle_type_t type, const char *label) {
    bool arrow_forward = (type == PARTICLE_FERMION);
    return add_propagator_internal(diagram, from, to, type, label, arrow_forward);
}

int feynman_add_incoming(feynman_diagram_t *diagram, int vertex, particle_type_t type,
                         const char *label, double direction) {
    if (!diagram || vertex < 0 || vertex >= diagram->num_vertices) {
        return -1;
    }

    /* Create external vertex */
    double rad = direction * PI / 180.0;
    double ext_x = diagram->vertices[vertex].x - 1.5 * cos(rad);
    double ext_y = diagram->vertices[vertex].y - 1.5 * sin(rad);

    int ext_v = feynman_add_external_vertex(diagram, ext_x, ext_y, label);
    if (ext_v < 0) return -1;

    /* Add propagator from external to vertex */
    return add_propagator_internal(diagram, ext_v, vertex, type, label, true);
}

int feynman_add_outgoing(feynman_diagram_t *diagram, int vertex, particle_type_t type,
                         const char *label, double direction) {
    if (!diagram || vertex < 0 || vertex >= diagram->num_vertices) {
        return -1;
    }

    /* Create external vertex */
    double rad = direction * PI / 180.0;
    double ext_x = diagram->vertices[vertex].x + 1.5 * cos(rad);
    double ext_y = diagram->vertices[vertex].y + 1.5 * sin(rad);

    int ext_v = feynman_add_external_vertex(diagram, ext_x, ext_y, label);
    if (ext_v < 0) return -1;

    /* Add propagator from vertex to external */
    return add_propagator_internal(diagram, vertex, ext_v, type, label, true);
}

/* ============================================================================
 * STANDARD DIAGRAMS
 * ============================================================================ */

feynman_diagram_t *feynman_create_qed_vertex(void) {
    feynman_diagram_t *fd = feynman_create("QED vertex");

    int v1 = feynman_add_external_vertex(fd, -2, 0, "e-");
    int v2 = feynman_add_vertex(fd, 0, 0);
    int v3 = feynman_add_external_vertex(fd, 2, 0, "e-");
    int v4 = feynman_add_external_vertex(fd, 0, 1.5, "gamma");

    feynman_add_fermion(fd, v1, v2, "e-");
    feynman_add_fermion(fd, v2, v3, "e-");
    feynman_add_photon(fd, v2, v4, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_ee_to_mumu(void) {
    feynman_diagram_t *fd = feynman_create("e+ e- -> mu+ mu-");

    int v1 = feynman_add_external_vertex(fd, -2, 1, "e-");
    int v2 = feynman_add_external_vertex(fd, -2, -1, "e+");
    int v3 = feynman_add_vertex(fd, 0, 0);
    int v4 = feynman_add_external_vertex(fd, 2, 1, "mu-");
    int v5 = feynman_add_external_vertex(fd, 2, -1, "mu+");

    feynman_add_fermion(fd, v1, v3, "e-");
    feynman_add_antifermion(fd, v2, v3, "e+");
    feynman_add_fermion(fd, v3, v4, "mu-");
    feynman_add_antifermion(fd, v3, v5, "mu+");

    return fd;
}

feynman_diagram_t *feynman_create_compton(void) {
    feynman_diagram_t *fd = feynman_create("Compton scattering");

    int v1 = feynman_add_external_vertex(fd, -2, 0, "e-");
    int v2 = feynman_add_external_vertex(fd, -1, 1.5, "gamma");
    int v3 = feynman_add_vertex(fd, -0.5, 0);
    int v4 = feynman_add_vertex(fd, 0.5, 0);
    int v5 = feynman_add_external_vertex(fd, 2, 0, "e-");
    int v6 = feynman_add_external_vertex(fd, 1, 1.5, "gamma");

    feynman_add_fermion(fd, v1, v3, "e-");
    feynman_add_photon(fd, v2, v3, "gamma");
    feynman_add_fermion(fd, v3, v4, "e-");
    feynman_add_fermion(fd, v4, v5, "e-");
    feynman_add_photon(fd, v4, v6, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_pair_annihilation(void) {
    feynman_diagram_t *fd = feynman_create("Pair annihilation");

    int v1 = feynman_add_external_vertex(fd, -2, 1, "e-");
    int v2 = feynman_add_external_vertex(fd, -2, -1, "e+");
    int v3 = feynman_add_vertex(fd, -0.5, 0);
    int v4 = feynman_add_vertex(fd, 0.5, 0);
    int v5 = feynman_add_external_vertex(fd, 2, 1, "gamma");
    int v6 = feynman_add_external_vertex(fd, 2, -1, "gamma");

    feynman_add_fermion(fd, v1, v3, "e-");
    feynman_add_antifermion(fd, v2, v3, "e+");
    feynman_add_fermion(fd, v3, v4, "");
    feynman_add_photon(fd, v4, v5, "gamma");
    feynman_add_photon(fd, v4, v6, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_electron_self_energy(void) {
    feynman_diagram_t *fd = feynman_create("Electron self-energy");
    fd->loop_order = 1;

    int v1 = feynman_add_external_vertex(fd, -2.5, 0, "e-");
    int v2 = feynman_add_vertex(fd, -1, 0);
    int v3 = feynman_add_vertex(fd, 0, 0.8);  /* Loop top */
    int v4 = feynman_add_vertex(fd, 1, 0);
    int v5 = feynman_add_external_vertex(fd, 2.5, 0, "e-");

    feynman_add_fermion(fd, v1, v2, "e-");
    feynman_add_fermion(fd, v2, v3, "");
    feynman_add_fermion(fd, v3, v4, "");
    feynman_add_fermion(fd, v4, v5, "e-");
    feynman_add_photon(fd, v2, v4, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_vacuum_polarization(void) {
    feynman_diagram_t *fd = feynman_create("Vacuum polarization");
    fd->loop_order = 1;

    int v1 = feynman_add_external_vertex(fd, -2.5, 0, "gamma");
    int v2 = feynman_add_vertex(fd, -1, 0);
    int v3 = feynman_add_vertex(fd, 0, 0.8);   /* Loop top */
    int v4 = feynman_add_vertex(fd, 0, -0.8);  /* Loop bottom */
    int v5 = feynman_add_vertex(fd, 1, 0);
    int v6 = feynman_add_external_vertex(fd, 2.5, 0, "gamma");

    feynman_add_photon(fd, v1, v2, "gamma");
    feynman_add_fermion(fd, v2, v3, "e-");
    feynman_add_fermion(fd, v3, v5, "e-");
    feynman_add_fermion(fd, v5, v4, "e+");
    feynman_add_fermion(fd, v4, v2, "e+");
    feynman_add_photon(fd, v5, v6, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_moller_scattering(void) {
    feynman_diagram_t *fd = feynman_create("Moller scattering (t-channel)");

    int v1 = feynman_add_external_vertex(fd, -2, 1.5, "e-");
    int v2 = feynman_add_external_vertex(fd, -2, -1.5, "e-");
    int v3 = feynman_add_vertex(fd, 0, 1);
    int v4 = feynman_add_vertex(fd, 0, -1);
    int v5 = feynman_add_external_vertex(fd, 2, 1.5, "e-");
    int v6 = feynman_add_external_vertex(fd, 2, -1.5, "e-");

    feynman_add_fermion(fd, v1, v3, "e-");
    feynman_add_fermion(fd, v3, v5, "e-");
    feynman_add_fermion(fd, v2, v4, "e-");
    feynman_add_fermion(fd, v4, v6, "e-");
    feynman_add_photon(fd, v3, v4, "gamma");

    return fd;
}

feynman_diagram_t *feynman_create_bhabha_scattering(void) {
    feynman_diagram_t *fd = feynman_create("Bhabha scattering (s-channel)");

    int v1 = feynman_add_external_vertex(fd, -2, 1, "e-");
    int v2 = feynman_add_external_vertex(fd, -2, -1, "e+");
    int v3 = feynman_add_vertex(fd, 0, 0);
    int v4 = feynman_add_external_vertex(fd, 2, 1, "e-");
    int v5 = feynman_add_external_vertex(fd, 2, -1, "e+");

    feynman_add_fermion(fd, v1, v3, "e-");
    feynman_add_antifermion(fd, v2, v3, "e+");
    feynman_add_fermion(fd, v3, v4, "e-");
    feynman_add_antifermion(fd, v3, v5, "e+");

    return fd;
}

/* ============================================================================
 * RENDERING OPTIONS
 * ============================================================================ */

feynman_options_t feynman_default_options(void) {
    feynman_options_t opts = {
        .width = 60,
        .height = 25,
        .show_labels = true,
        .show_momentum = false,
        .show_vertices = true,
        .wave_amplitude = 3,
        .wave_frequency = 4,
        .font_size = 12
    };
    strcpy(opts.font_family, "monospace");
    return opts;
}

feynman_options_t feynman_publication_options(void) {
    feynman_options_t opts = feynman_default_options();
    opts.width = 80;
    opts.height = 30;
    opts.font_size = 14;
    strcpy(opts.font_family, "Computer Modern");
    return opts;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const feynman_vertex_t *feynman_get_vertex(const feynman_diagram_t *diagram, int id) {
    if (!diagram || id < 0 || id >= diagram->num_vertices) {
        return NULL;
    }
    return &diagram->vertices[id];
}

void feynman_update_bounds(feynman_diagram_t *diagram) {
    if (!diagram || diagram->num_vertices == 0) return;

    diagram->min_x = diagram->max_x = diagram->vertices[0].x;
    diagram->min_y = diagram->max_y = diagram->vertices[0].y;

    for (int i = 1; i < diagram->num_vertices; i++) {
        double x = diagram->vertices[i].x;
        double y = diagram->vertices[i].y;
        if (x < diagram->min_x) diagram->min_x = x;
        if (x > diagram->max_x) diagram->max_x = x;
        if (y < diagram->min_y) diagram->min_y = y;
        if (y > diagram->max_y) diagram->max_y = y;
    }
}

const char *feynman_particle_type_name(particle_type_t type) {
    switch (type) {
        case PARTICLE_FERMION: return "fermion";
        case PARTICLE_ANTIFERMION: return "antifermion";
        case PARTICLE_PHOTON: return "photon";
        case PARTICLE_GLUON: return "gluon";
        case PARTICLE_W_BOSON: return "W boson";
        case PARTICLE_Z_BOSON: return "Z boson";
        case PARTICLE_HIGGS: return "Higgs";
        case PARTICLE_SCALAR: return "scalar";
        case PARTICLE_GHOST: return "ghost";
        case PARTICLE_GRAVITON: return "graviton";
        default: return "unknown";
    }
}

const char *feynman_tikz_style(particle_type_t type) {
    switch (type) {
        case PARTICLE_FERMION: return "fermion";
        case PARTICLE_ANTIFERMION: return "anti fermion";
        case PARTICLE_PHOTON: return "photon";
        case PARTICLE_GLUON: return "gluon";
        case PARTICLE_W_BOSON:
        case PARTICLE_Z_BOSON: return "boson";
        case PARTICLE_HIGGS:
        case PARTICLE_SCALAR: return "scalar";
        case PARTICLE_GHOST: return "ghost";
        case PARTICLE_GRAVITON: return "graviton";
        default: return "plain";
    }
}

/* ============================================================================
 * ASCII RENDERING
 * ============================================================================ */

typedef struct {
    char **grid;
    int width;
    int height;
} ascii_grid_t;

static ascii_grid_t *create_grid(int width, int height) {
    ascii_grid_t *grid = malloc(sizeof(ascii_grid_t));
    if (!grid) return NULL;

    grid->width = width;
    grid->height = height;
    grid->grid = malloc(height * sizeof(char *));
    if (!grid->grid) {
        free(grid);
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        grid->grid[i] = malloc(width + 1);
        if (!grid->grid[i]) {
            for (int j = 0; j < i; j++) free(grid->grid[j]);
            free(grid->grid);
            free(grid);
            return NULL;
        }
        memset(grid->grid[i], ' ', width);
        grid->grid[i][width] = '\0';
    }

    return grid;
}

static void free_grid(ascii_grid_t *grid) {
    if (grid) {
        for (int i = 0; i < grid->height; i++) {
            free(grid->grid[i]);
        }
        free(grid->grid);
        free(grid);
    }
}

static char *grid_to_string(ascii_grid_t *grid) {
    size_t total = 0;
    for (int i = 0; i < grid->height; i++) {
        total += strlen(grid->grid[i]) + 1;
    }
    total++;

    char *result = malloc(total);
    if (!result) return NULL;

    char *ptr = result;
    for (int i = 0; i < grid->height; i++) {
        int len = strlen(grid->grid[i]);
        while (len > 0 && grid->grid[i][len - 1] == ' ') len--;
        memcpy(ptr, grid->grid[i], len);
        ptr += len;
        *ptr++ = '\n';
    }
    *ptr = '\0';

    return result;
}

static void draw_char(ascii_grid_t *grid, int x, int y, char c) {
    if (x >= 0 && x < grid->width && y >= 0 && y < grid->height) {
        grid->grid[y][x] = c;
    }
}

static void draw_text(ascii_grid_t *grid, int x, int y, const char *text) {
    int len = strlen(text);
    for (int i = 0; i < len; i++) {
        draw_char(grid, x + i, y, text[i]);
    }
}

static void draw_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2, char c) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (1) {
        draw_char(grid, x1, y1, c);
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

static void draw_fermion_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2, bool forward) {
    int dx = x2 - x1;
    int dy = y2 - y1;

    if (abs(dx) > abs(dy)) {
        /* More horizontal than vertical */
        char c = (forward) ? '>' : '<';
        if (dx < 0) c = (forward) ? '<' : '>';
        for (int x = x1; x != x2; x += (dx > 0 ? 1 : -1)) {
            int y = y1 + (x - x1) * dy / (dx != 0 ? dx : 1);
            draw_char(grid, x, y, '-');
        }
        /* Arrow at midpoint */
        int mx = (x1 + x2) / 2;
        int my = y1 + (mx - x1) * dy / (dx != 0 ? dx : 1);
        draw_char(grid, mx, my, c);
    } else {
        /* More vertical */
        char c = (forward) ? 'v' : '^';
        if (dy < 0) c = (forward) ? '^' : 'v';
        for (int y = y1; y != y2; y += (dy > 0 ? 1 : -1)) {
            int x = x1 + (y - y1) * dx / (dy != 0 ? dy : 1);
            draw_char(grid, x, y, '|');
        }
        int my = (y1 + y2) / 2;
        int mx = x1 + (my - y1) * dx / (dy != 0 ? dy : 1);
        draw_char(grid, mx, my, c);
    }
}

static void draw_wavy_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int steps = (dx > dy) ? dx : dy;
    if (steps == 0) steps = 1;

    for (int i = 0; i <= steps; i++) {
        int x = x1 + (x2 - x1) * i / steps;
        int y = y1 + (y2 - y1) * i / steps;
        char c = (i % 2 == 0) ? '~' : '~';
        draw_char(grid, x, y, c);
    }
}

static void draw_curly_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int steps = (dx > dy) ? dx : dy;
    if (steps == 0) steps = 1;

    for (int i = 0; i <= steps; i++) {
        int x = x1 + (x2 - x1) * i / steps;
        int y = y1 + (y2 - y1) * i / steps;
        char c = (i % 3 == 0) ? '@' : ((i % 3 == 1) ? 'c' : '@');
        draw_char(grid, x, y, c);
    }
}

static void draw_dashed_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int steps = (dx > dy) ? dx : dy;
    if (steps == 0) steps = 1;

    for (int i = 0; i <= steps; i++) {
        if (i % 2 == 0) {
            int x = x1 + (x2 - x1) * i / steps;
            int y = y1 + (y2 - y1) * i / steps;
            draw_char(grid, x, y, '-');
        }
    }
}

static void draw_dotted_line(ascii_grid_t *grid, int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int steps = (dx > dy) ? dx : dy;
    if (steps == 0) steps = 1;

    for (int i = 0; i <= steps; i++) {
        if (i % 3 == 0) {
            int x = x1 + (x2 - x1) * i / steps;
            int y = y1 + (y2 - y1) * i / steps;
            draw_char(grid, x, y, '.');
        }
    }
}

char *feynman_render_ascii(const feynman_diagram_t *diagram, const feynman_options_t *opts) {
    if (!diagram) return NULL;

    feynman_options_t default_opts = feynman_default_options();
    if (!opts) opts = &default_opts;

    int width = opts->width;
    int height = opts->height;

    ascii_grid_t *grid = create_grid(width, height);
    if (!grid) return NULL;

    /* Calculate scaling */
    double x_range = diagram->max_x - diagram->min_x;
    double y_range = diagram->max_y - diagram->min_y;
    if (x_range < 0.001) x_range = 1;
    if (y_range < 0.001) y_range = 1;

    int margin = 8;
    double scale_x = (width - 2 * margin) / x_range;
    double scale_y = (height - 2 * margin) / y_range;

    /* Map coordinate helper */
    #define MAP_X(x) ((int)(margin + ((x) - diagram->min_x) * scale_x))
    #define MAP_Y(y) ((int)(height - margin - 1 - ((y) - diagram->min_y) * scale_y))

    /* Draw title/process */
    if (diagram->process[0] != '\0') {
        draw_text(grid, (width - strlen(diagram->process)) / 2, 1, diagram->process);
    }

    /* Draw propagators */
    for (int i = 0; i < diagram->num_propagators; i++) {
        const feynman_propagator_t *p = &diagram->propagators[i];
        const feynman_vertex_t *v1 = &diagram->vertices[p->from_vertex];
        const feynman_vertex_t *v2 = &diagram->vertices[p->to_vertex];

        int x1 = MAP_X(v1->x);
        int y1 = MAP_Y(v1->y);
        int x2 = MAP_X(v2->x);
        int y2 = MAP_Y(v2->y);

        switch (p->type) {
            case PARTICLE_FERMION:
                draw_fermion_line(grid, x1, y1, x2, y2, true);
                break;
            case PARTICLE_ANTIFERMION:
                draw_fermion_line(grid, x1, y1, x2, y2, false);
                break;
            case PARTICLE_PHOTON:
                draw_wavy_line(grid, x1, y1, x2, y2);
                break;
            case PARTICLE_GLUON:
                draw_curly_line(grid, x1, y1, x2, y2);
                break;
            case PARTICLE_W_BOSON:
            case PARTICLE_Z_BOSON:
            case PARTICLE_SCALAR:
                draw_dashed_line(grid, x1, y1, x2, y2);
                break;
            case PARTICLE_HIGGS:
                draw_dotted_line(grid, x1, y1, x2, y2);
                break;
            default:
                draw_line(grid, x1, y1, x2, y2, '-');
        }

        /* Draw label at midpoint */
        if (opts->show_labels && p->label[0] != '\0') {
            int mx = (x1 + x2) / 2;
            int my = (y1 + y2) / 2 - 1;
            if (my < 0) my = 0;
            draw_text(grid, mx - strlen(p->label) / 2, my, p->label);
        }
    }

    /* Draw vertices */
    if (opts->show_vertices) {
        for (int i = 0; i < diagram->num_vertices; i++) {
            const feynman_vertex_t *v = &diagram->vertices[i];
            int x = MAP_X(v->x);
            int y = MAP_Y(v->y);

            if (!v->is_external) {
                draw_char(grid, x, y, 'o');
            } else if (v->label[0] != '\0') {
                draw_text(grid, x - strlen(v->label) / 2, y, v->label);
            }
        }
    }

    #undef MAP_X
    #undef MAP_Y

    char *result = grid_to_string(grid);
    free_grid(grid);
    return result;
}

void feynman_print_ascii(const feynman_diagram_t *diagram) {
    char *ascii = feynman_render_ascii(diagram, NULL);
    if (ascii) {
        printf("%s", ascii);
        free(ascii);
    }
}

/* ============================================================================
 * SVG RENDERING
 * ============================================================================ */

char *feynman_render_svg(const feynman_diagram_t *diagram, int width, int height) {
    if (!diagram) return NULL;

    string_builder_t *sb = sb_create();
    if (!sb) return NULL;

    /* Calculate scaling */
    double x_range = diagram->max_x - diagram->min_x;
    double y_range = diagram->max_y - diagram->min_y;
    if (x_range < 0.001) x_range = 1;
    if (y_range < 0.001) y_range = 1;

    int margin = 50;
    double scale_x = (width - 2 * margin) / x_range;
    double scale_y = (height - 2 * margin) / y_range;
    double scale = (scale_x < scale_y) ? scale_x : scale_y;

    /* SVG header */
    sb_appendf(sb, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    sb_appendf(sb, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
               width, height, width, height);

    /* Styles */
    sb_append(sb, "<defs>\n");
    sb_append(sb, "  <style>\n");
    sb_append(sb, "    .fermion { stroke: #333; stroke-width: 2; fill: none; }\n");
    sb_append(sb, "    .photon { stroke: #333; stroke-width: 2; fill: none; }\n");
    sb_append(sb, "    .gluon { stroke: #333; stroke-width: 2; fill: none; }\n");
    sb_append(sb, "    .boson { stroke: #333; stroke-width: 2; stroke-dasharray: 5,5; fill: none; }\n");
    sb_append(sb, "    .scalar { stroke: #333; stroke-width: 2; stroke-dasharray: 2,2; fill: none; }\n");
    sb_append(sb, "    .vertex { fill: #333; }\n");
    sb_append(sb, "    .label { font-family: 'Times New Roman', serif; font-size: 14px; font-style: italic; }\n");
    sb_append(sb, "    .title { font-family: 'Times New Roman', serif; font-size: 16px; text-anchor: middle; }\n");
    sb_append(sb, "  </style>\n");
    sb_append(sb, "  <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" ");
    sb_append(sb, "orient=\"auto\" markerUnits=\"strokeWidth\">\n");
    sb_append(sb, "    <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#333\"/>\n");
    sb_append(sb, "  </marker>\n");
    sb_append(sb, "</defs>\n");

    /* Background */
    sb_appendf(sb, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", width, height);

    /* Title */
    if (diagram->process[0] != '\0') {
        sb_appendf(sb, "<text x=\"%d\" y=\"25\" class=\"title\">%s</text>\n",
                   width / 2, diagram->process);
    }

    /* Map coordinates */
    #define SVG_X(x) (margin + ((x) - diagram->min_x) * scale)
    #define SVG_Y(y) (height - margin - ((y) - diagram->min_y) * scale)

    /* Draw propagators */
    for (int i = 0; i < diagram->num_propagators; i++) {
        const feynman_propagator_t *p = &diagram->propagators[i];
        const feynman_vertex_t *v1 = &diagram->vertices[p->from_vertex];
        const feynman_vertex_t *v2 = &diagram->vertices[p->to_vertex];

        double x1 = SVG_X(v1->x);
        double y1 = SVG_Y(v1->y);
        double x2 = SVG_X(v2->x);
        double y2 = SVG_Y(v2->y);

        switch (p->type) {
            case PARTICLE_FERMION:
                sb_appendf(sb, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                           "class=\"fermion\" marker-end=\"url(#arrow)\"/>\n",
                           x1, y1, x2, y2);
                break;

            case PARTICLE_ANTIFERMION:
                sb_appendf(sb, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                           "class=\"fermion\" marker-end=\"url(#arrow)\"/>\n",
                           x2, y2, x1, y1);  /* Reversed direction */
                break;

            case PARTICLE_PHOTON: {
                /* Wavy line using sine wave path */
                double length = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                int waves = (int)(length / 15);
                if (waves < 2) waves = 2;
                double angle = atan2(y2 - y1, x2 - x1);
                double perp_x = -sin(angle);
                double perp_y = cos(angle);
                double amp = 8;

                sb_appendf(sb, "<path d=\"M %.1f %.1f", x1, y1);
                for (int w = 0; w <= waves * 10; w++) {
                    double t = (double)w / (waves * 10);
                    double px = x1 + (x2 - x1) * t;
                    double py = y1 + (y2 - y1) * t;
                    double offset = amp * sin(t * waves * 2 * PI);
                    sb_appendf(sb, " L %.1f %.1f", px + perp_x * offset, py + perp_y * offset);
                }
                sb_append(sb, "\" class=\"photon\"/>\n");
                break;
            }

            case PARTICLE_GLUON: {
                /* Curly line */
                double length = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                int loops = (int)(length / 20);
                if (loops < 2) loops = 2;
                double angle = atan2(y2 - y1, x2 - x1);
                double perp_x = -sin(angle);
                double perp_y = cos(angle);
                double amp = 10;

                sb_appendf(sb, "<path d=\"M %.1f %.1f", x1, y1);
                for (int w = 0; w <= loops * 20; w++) {
                    double t = (double)w / (loops * 20);
                    double px = x1 + (x2 - x1) * t;
                    double py = y1 + (y2 - y1) * t;
                    double phase = t * loops * 2 * PI;
                    double offset = amp * sin(phase) * (1 + 0.5 * sin(phase));
                    sb_appendf(sb, " L %.1f %.1f", px + perp_x * offset, py + perp_y * offset);
                }
                sb_append(sb, "\" class=\"gluon\"/>\n");
                break;
            }

            case PARTICLE_W_BOSON:
            case PARTICLE_Z_BOSON:
                sb_appendf(sb, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" class=\"boson\"/>\n",
                           x1, y1, x2, y2);
                break;

            case PARTICLE_HIGGS:
            case PARTICLE_SCALAR:
                sb_appendf(sb, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" class=\"scalar\"/>\n",
                           x1, y1, x2, y2);
                break;

            default:
                sb_appendf(sb, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" class=\"fermion\"/>\n",
                           x1, y1, x2, y2);
        }

        /* Label */
        if (p->label[0] != '\0') {
            double mx = (x1 + x2) / 2;
            double my = (y1 + y2) / 2 - 8;
            sb_appendf(sb, "<text x=\"%.1f\" y=\"%.1f\" class=\"label\">%s</text>\n",
                       mx, my, p->label);
        }
    }

    /* Draw vertices */
    for (int i = 0; i < diagram->num_vertices; i++) {
        const feynman_vertex_t *v = &diagram->vertices[i];
        double x = SVG_X(v->x);
        double y = SVG_Y(v->y);

        if (!v->is_external) {
            sb_appendf(sb, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"4\" class=\"vertex\"/>\n", x, y);
        }

        if (v->label[0] != '\0' && v->is_external) {
            /* Offset label from vertex */
            double lx = x;
            double ly = y;
            if (v->x < (diagram->min_x + diagram->max_x) / 2) {
                lx -= 25;
            } else {
                lx += 10;
            }
            sb_appendf(sb, "<text x=\"%.1f\" y=\"%.1f\" class=\"label\">%s</text>\n",
                       lx, ly, v->label);
        }
    }

    #undef SVG_X
    #undef SVG_Y

    sb_append(sb, "</svg>\n");

    char *result = sb->buffer;
    sb->buffer = NULL;
    sb_free(sb);
    return result;
}

int feynman_save_svg(const feynman_diagram_t *diagram, const char *filename,
                     int width, int height) {
    char *svg = feynman_render_svg(diagram, width, height);
    if (!svg) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) {
        free(svg);
        return -1;
    }

    fputs(svg, f);
    fclose(f);
    free(svg);
    return 0;
}

/* ============================================================================
 * LATEX RENDERING
 * ============================================================================ */

char *feynman_render_latex(const feynman_diagram_t *diagram) {
    if (!diagram) return NULL;

    string_builder_t *sb = sb_create();
    if (!sb) return NULL;

    sb_append(sb, "% Requires: \\usepackage{tikz-feynman}\n");
    sb_append(sb, "\\begin{tikzpicture}\n");
    sb_append(sb, "  \\begin{feynman}\n");

    /* Vertices */
    for (int i = 0; i < diagram->num_vertices; i++) {
        const feynman_vertex_t *v = &diagram->vertices[i];
        if (v->label[0] != '\0' && v->is_external) {
            sb_appendf(sb, "    \\vertex (v%d) at (%.1f, %.1f) {$%s$};\n",
                       i, v->x, v->y, v->label);
        } else {
            sb_appendf(sb, "    \\vertex (v%d) at (%.1f, %.1f);\n",
                       i, v->x, v->y);
        }
    }

    sb_append(sb, "\n    \\diagram* {\n");

    /* Propagators */
    for (int i = 0; i < diagram->num_propagators; i++) {
        const feynman_propagator_t *p = &diagram->propagators[i];
        const char *style = feynman_tikz_style(p->type);

        if (p->label[0] != '\0') {
            sb_appendf(sb, "      (v%d) -- [%s, edge label=$%s$] (v%d),\n",
                       p->from_vertex, style, p->label, p->to_vertex);
        } else {
            sb_appendf(sb, "      (v%d) -- [%s] (v%d),\n",
                       p->from_vertex, style, p->to_vertex);
        }
    }

    sb_append(sb, "    };\n");
    sb_append(sb, "  \\end{feynman}\n");
    sb_append(sb, "\\end{tikzpicture}");

    char *result = sb->buffer;
    sb->buffer = NULL;
    sb_free(sb);
    return result;
}

int feynman_save_latex(const feynman_diagram_t *diagram, const char *filename) {
    char *latex = feynman_render_latex(diagram);
    if (!latex) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) {
        free(latex);
        return -1;
    }

    fputs(latex, f);
    fclose(f);
    free(latex);
    return 0;
}
