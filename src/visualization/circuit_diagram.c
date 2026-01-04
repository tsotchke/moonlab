/**
 * @file circuit_diagram.c
 * @brief Quantum Circuit Diagram Renderer Implementation
 */

#include "circuit_diagram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* ============================================================================
 * INTERNAL CONSTANTS
 * ============================================================================ */

#define INITIAL_CAPACITY 64
#define ASCII_GATE_WIDTH 5
#define ASCII_WIRE_CHAR '-'
#define ASCII_CLASSICAL_WIRE '='
#define PI 3.14159265358979323846

/* Box drawing characters (UTF-8) */
static const char *BOX_TL = "\xe2\x94\x8c";  /* top-left corner */
static const char *BOX_TR = "\xe2\x94\x90";  /* top-right corner */
static const char *BOX_BL = "\xe2\x94\x94";  /* bottom-left corner */
static const char *BOX_BR = "\xe2\x94\x98";  /* bottom-right corner */
static const char *BOX_H = "\xe2\x94\x80";   /* horizontal */
static const char *BOX_V = "\xe2\x94\x82";   /* vertical */
static const char *CTRL_DOT = "\xe2\x97\x8f"; /* filled circle for control */
static const char *OPLUS = "\xe2\x8a\x95";    /* circled plus for target */
static const char *CROSS = "\xc3\x97";        /* multiplication sign for SWAP */
static const char *METER = "M";               /* measurement symbol */

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static int max_int(int a, int b) {
    return a > b ? a : b;
}

static int min_int(int a, int b) {
    return a < b ? a : b;
}

/* ============================================================================
 * CIRCUIT CREATION AND MANAGEMENT
 * ============================================================================ */

circuit_diagram_t *circuit_create(int num_qubits) {
    return circuit_create_with_classical(num_qubits, num_qubits);
}

circuit_diagram_t *circuit_create_with_classical(int num_qubits, int num_classical) {
    if (num_qubits < 1 || num_qubits > CIRCUIT_MAX_QUBITS) {
        return NULL;
    }
    if (num_classical < 0 || num_classical > CIRCUIT_MAX_CLASSICAL) {
        return NULL;
    }

    circuit_diagram_t *circuit = calloc(1, sizeof(circuit_diagram_t));
    if (!circuit) return NULL;

    circuit->elements = calloc(INITIAL_CAPACITY, sizeof(circuit_element_t));
    if (!circuit->elements) {
        free(circuit);
        return NULL;
    }

    circuit->num_qubits = num_qubits;
    circuit->num_classical = num_classical;
    circuit->num_elements = 0;
    circuit->capacity = INITIAL_CAPACITY;
    circuit->title[0] = '\0';

    /* Initialize default qubit labels */
    for (int i = 0; i < num_qubits; i++) {
        snprintf(circuit->qubit_labels[i], CIRCUIT_MAX_LABEL_LEN, "q%d", i);
        circuit->current_time[i] = 0;
    }

    return circuit;
}

void circuit_free(circuit_diagram_t *circuit) {
    if (circuit) {
        free(circuit->elements);
        free(circuit);
    }
}

void circuit_set_title(circuit_diagram_t *circuit, const char *title) {
    if (circuit && title) {
        strncpy(circuit->title, title, CIRCUIT_MAX_TITLE_LEN - 1);
        circuit->title[CIRCUIT_MAX_TITLE_LEN - 1] = '\0';
    }
}

void circuit_set_qubit_label(circuit_diagram_t *circuit, int qubit, const char *label) {
    if (circuit && qubit >= 0 && qubit < circuit->num_qubits && label) {
        strncpy(circuit->qubit_labels[qubit], label, CIRCUIT_MAX_LABEL_LEN - 1);
        circuit->qubit_labels[qubit][CIRCUIT_MAX_LABEL_LEN - 1] = '\0';
    }
}

int circuit_get_depth(const circuit_diagram_t *circuit) {
    if (!circuit) return 0;
    int max_time = 0;
    for (int i = 0; i < circuit->num_qubits; i++) {
        if (circuit->current_time[i] > max_time) {
            max_time = circuit->current_time[i];
        }
    }
    return max_time;
}

void circuit_clear(circuit_diagram_t *circuit) {
    if (circuit) {
        circuit->num_elements = 0;
        for (int i = 0; i < circuit->num_qubits; i++) {
            circuit->current_time[i] = 0;
        }
    }
}

/* ============================================================================
 * INTERNAL: ADD ELEMENT
 * ============================================================================ */

static int ensure_capacity(circuit_diagram_t *circuit) {
    if (circuit->num_elements >= circuit->capacity) {
        int new_capacity = circuit->capacity * 2;
        circuit_element_t *new_elements = realloc(circuit->elements,
                                                  new_capacity * sizeof(circuit_element_t));
        if (!new_elements) return -1;
        circuit->elements = new_elements;
        circuit->capacity = new_capacity;
    }
    return 0;
}

static circuit_element_t *add_element(circuit_diagram_t *circuit) {
    if (ensure_capacity(circuit) < 0) return NULL;
    return &circuit->elements[circuit->num_elements++];
}

/* ============================================================================
 * ADDING CIRCUIT ELEMENTS
 * ============================================================================ */

int circuit_add_gate(circuit_diagram_t *circuit, int qubit, const char *name) {
    if (!circuit || qubit < 0 || qubit >= circuit->num_qubits || !name) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    elem->type = CIRCUIT_GATE_SINGLE;
    elem->qubit = qubit;
    elem->target_qubit = -1;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = circuit->current_time[qubit];
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, name, CIRCUIT_MAX_LABEL_LEN - 1);
    elem->name[CIRCUIT_MAX_LABEL_LEN - 1] = '\0';

    circuit->current_time[qubit]++;
    return elem->time_slot;
}

int circuit_add_rotation(circuit_diagram_t *circuit, int qubit, const char *name, double angle) {
    if (!circuit || qubit < 0 || qubit >= circuit->num_qubits || !name) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    elem->type = CIRCUIT_GATE_SINGLE;
    elem->qubit = qubit;
    elem->target_qubit = -1;
    elem->control_qubit2 = -1;
    elem->angle = angle;
    elem->time_slot = circuit->current_time[qubit];
    elem->classical_bit = -1;
    elem->color = 0;

    /* Format gate name with angle */
    char angle_str[16];
    circuit_format_angle(angle, angle_str, sizeof(angle_str));
    snprintf(elem->name, CIRCUIT_MAX_LABEL_LEN, "%s(%s)", name, angle_str);

    circuit->current_time[qubit]++;
    return elem->time_slot;
}

int circuit_add_controlled(circuit_diagram_t *circuit, int control, int target, const char *name) {
    if (!circuit || control < 0 || control >= circuit->num_qubits ||
        target < 0 || target >= circuit->num_qubits || control == target || !name) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    /* Find common time slot */
    int time_slot = max_int(circuit->current_time[control], circuit->current_time[target]);

    elem->type = CIRCUIT_GATE_CONTROLLED;
    elem->qubit = control;
    elem->target_qubit = target;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, name, CIRCUIT_MAX_LABEL_LEN - 1);
    elem->name[CIRCUIT_MAX_LABEL_LEN - 1] = '\0';

    circuit->current_time[control] = time_slot + 1;
    circuit->current_time[target] = time_slot + 1;
    return time_slot;
}

int circuit_add_controlled_rotation(circuit_diagram_t *circuit, int control, int target,
                                    const char *name, double angle) {
    if (!circuit || control < 0 || control >= circuit->num_qubits ||
        target < 0 || target >= circuit->num_qubits || control == target || !name) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    int time_slot = max_int(circuit->current_time[control], circuit->current_time[target]);

    elem->type = CIRCUIT_GATE_CONTROLLED;
    elem->qubit = control;
    elem->target_qubit = target;
    elem->control_qubit2 = -1;
    elem->angle = angle;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;

    char angle_str[16];
    circuit_format_angle(angle, angle_str, sizeof(angle_str));
    snprintf(elem->name, CIRCUIT_MAX_LABEL_LEN, "%s(%s)", name, angle_str);

    circuit->current_time[control] = time_slot + 1;
    circuit->current_time[target] = time_slot + 1;
    return time_slot;
}

int circuit_add_toffoli(circuit_diagram_t *circuit, int ctrl1, int ctrl2, int target) {
    if (!circuit || ctrl1 < 0 || ctrl1 >= circuit->num_qubits ||
        ctrl2 < 0 || ctrl2 >= circuit->num_qubits ||
        target < 0 || target >= circuit->num_qubits ||
        ctrl1 == ctrl2 || ctrl1 == target || ctrl2 == target) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    int time_slot = max_int(circuit->current_time[ctrl1],
                           max_int(circuit->current_time[ctrl2], circuit->current_time[target]));

    elem->type = CIRCUIT_GATE_MULTI;
    elem->qubit = ctrl1;
    elem->target_qubit = target;
    elem->control_qubit2 = ctrl2;
    elem->angle = 0;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, "CCX", CIRCUIT_MAX_LABEL_LEN - 1);

    circuit->current_time[ctrl1] = time_slot + 1;
    circuit->current_time[ctrl2] = time_slot + 1;
    circuit->current_time[target] = time_slot + 1;
    return time_slot;
}

int circuit_add_fredkin(circuit_diagram_t *circuit, int control, int target1, int target2) {
    if (!circuit || control < 0 || control >= circuit->num_qubits ||
        target1 < 0 || target1 >= circuit->num_qubits ||
        target2 < 0 || target2 >= circuit->num_qubits ||
        control == target1 || control == target2 || target1 == target2) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    int time_slot = max_int(circuit->current_time[control],
                           max_int(circuit->current_time[target1], circuit->current_time[target2]));

    elem->type = CIRCUIT_GATE_MULTI;
    elem->qubit = control;
    elem->target_qubit = target1;
    elem->control_qubit2 = target2;
    elem->angle = 0;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, "CSWAP", CIRCUIT_MAX_LABEL_LEN - 1);

    circuit->current_time[control] = time_slot + 1;
    circuit->current_time[target1] = time_slot + 1;
    circuit->current_time[target2] = time_slot + 1;
    return time_slot;
}

int circuit_add_swap(circuit_diagram_t *circuit, int qubit1, int qubit2) {
    if (!circuit || qubit1 < 0 || qubit1 >= circuit->num_qubits ||
        qubit2 < 0 || qubit2 >= circuit->num_qubits || qubit1 == qubit2) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    int time_slot = max_int(circuit->current_time[qubit1], circuit->current_time[qubit2]);

    elem->type = CIRCUIT_GATE_MULTI;
    elem->qubit = qubit1;
    elem->target_qubit = qubit2;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, "SWAP", CIRCUIT_MAX_LABEL_LEN - 1);

    circuit->current_time[qubit1] = time_slot + 1;
    circuit->current_time[qubit2] = time_slot + 1;
    return time_slot;
}

int circuit_add_measurement(circuit_diagram_t *circuit, int qubit, int classical_bit) {
    if (!circuit || qubit < 0 || qubit >= circuit->num_qubits ||
        classical_bit < 0 || classical_bit >= circuit->num_classical) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    elem->type = CIRCUIT_MEASUREMENT;
    elem->qubit = qubit;
    elem->target_qubit = -1;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = circuit->current_time[qubit];
    elem->classical_bit = classical_bit;
    elem->color = 0;
    snprintf(elem->name, CIRCUIT_MAX_LABEL_LEN, "M%d", classical_bit);

    circuit->current_time[qubit]++;
    return elem->time_slot;
}

int circuit_add_barrier(circuit_diagram_t *circuit, const int *qubits, int num_qubits) {
    if (!circuit || !qubits || num_qubits < 1) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    /* Find max time across all qubits in barrier */
    int time_slot = 0;
    for (int i = 0; i < num_qubits; i++) {
        if (qubits[i] >= 0 && qubits[i] < circuit->num_qubits) {
            if (circuit->current_time[qubits[i]] > time_slot) {
                time_slot = circuit->current_time[qubits[i]];
            }
        }
    }

    elem->type = CIRCUIT_BARRIER;
    elem->qubit = qubits[0];
    elem->target_qubit = -1;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = time_slot;
    elem->classical_bit = -1;
    elem->color = 0;
    elem->name[0] = '\0';

    /* Copy barrier qubits */
    elem->num_barrier_qubits = min_int(num_qubits, CIRCUIT_MAX_QUBITS);
    for (int i = 0; i < elem->num_barrier_qubits; i++) {
        elem->barrier_qubits[i] = qubits[i];
    }

    /* Update time for all barrier qubits */
    for (int i = 0; i < num_qubits; i++) {
        if (qubits[i] >= 0 && qubits[i] < circuit->num_qubits) {
            circuit->current_time[qubits[i]] = time_slot + 1;
        }
    }

    return time_slot;
}

int circuit_add_barrier_all(circuit_diagram_t *circuit) {
    if (!circuit) return -1;

    int qubits[CIRCUIT_MAX_QUBITS];
    for (int i = 0; i < circuit->num_qubits; i++) {
        qubits[i] = i;
    }
    return circuit_add_barrier(circuit, qubits, circuit->num_qubits);
}

int circuit_add_reset(circuit_diagram_t *circuit, int qubit) {
    if (!circuit || qubit < 0 || qubit >= circuit->num_qubits) {
        return -1;
    }

    circuit_element_t *elem = add_element(circuit);
    if (!elem) return -1;

    elem->type = CIRCUIT_RESET;
    elem->qubit = qubit;
    elem->target_qubit = -1;
    elem->control_qubit2 = -1;
    elem->angle = 0;
    elem->time_slot = circuit->current_time[qubit];
    elem->classical_bit = -1;
    elem->color = 0;
    strncpy(elem->name, "|0>", CIRCUIT_MAX_LABEL_LEN - 1);

    circuit->current_time[qubit]++;
    return elem->time_slot;
}

/* ============================================================================
 * RENDERING OPTIONS
 * ============================================================================ */

render_options_t circuit_default_options(void) {
    render_options_t opts = {
        .gate_width = 5,
        .gate_height = 3,
        .wire_spacing = 3,
        .show_grid = false,
        .show_labels = true,
        .show_barriers = true,
        .color_phases = false,
        .compact = false,
        .font_family = "monospace",
        .font_size = 12,
        .svg_width = 800,
        .svg_height = 400
    };
    strcpy(opts.font_family, "monospace");
    return opts;
}

render_options_t circuit_compact_options(void) {
    render_options_t opts = circuit_default_options();
    opts.gate_width = 3;
    opts.gate_height = 1;
    opts.wire_spacing = 1;
    opts.compact = true;
    return opts;
}

render_options_t circuit_publication_options(void) {
    render_options_t opts = circuit_default_options();
    opts.gate_width = 7;
    opts.gate_height = 3;
    opts.wire_spacing = 4;
    opts.show_grid = true;
    opts.font_size = 14;
    strcpy(opts.font_family, "Computer Modern");
    return opts;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

char *circuit_format_angle(double angle, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size < 2) return buffer;

    /* Check for common pi fractions */
    double normalized = fmod(angle, 2 * PI);
    if (normalized < 0) normalized += 2 * PI;

    double pi_factor = angle / PI;

    /* Check common fractions */
    const struct { double factor; const char *str; } fractions[] = {
        {0.0, "0"},
        {0.25, "pi/4"},
        {0.5, "pi/2"},
        {0.75, "3pi/4"},
        {1.0, "pi"},
        {1.25, "5pi/4"},
        {1.5, "3pi/2"},
        {1.75, "7pi/4"},
        {2.0, "2pi"},
        {-0.25, "-pi/4"},
        {-0.5, "-pi/2"},
        {-0.75, "-3pi/4"},
        {-1.0, "-pi"},
        {1.0/3.0, "pi/3"},
        {2.0/3.0, "2pi/3"},
        {1.0/6.0, "pi/6"},
        {5.0/6.0, "5pi/6"},
    };

    for (size_t i = 0; i < sizeof(fractions) / sizeof(fractions[0]); i++) {
        if (fabs(pi_factor - fractions[i].factor) < 1e-10) {
            strncpy(buffer, fractions[i].str, buffer_size - 1);
            buffer[buffer_size - 1] = '\0';
            return buffer;
        }
    }

    /* Fall back to decimal */
    snprintf(buffer, buffer_size, "%.3f", angle);
    return buffer;
}

const circuit_element_t *circuit_get_element_at(const circuit_diagram_t *circuit,
                                                int qubit, int time) {
    if (!circuit) return NULL;

    for (int i = 0; i < circuit->num_elements; i++) {
        const circuit_element_t *elem = &circuit->elements[i];
        if (elem->time_slot == time) {
            if (elem->qubit == qubit) return elem;
            if (elem->target_qubit == qubit) return elem;
            if (elem->control_qubit2 == qubit) return elem;

            /* Check barrier qubits */
            if (elem->type == CIRCUIT_BARRIER) {
                for (int j = 0; j < elem->num_barrier_qubits; j++) {
                    if (elem->barrier_qubits[j] == qubit) return elem;
                }
            }
        }
    }
    return NULL;
}

/* ============================================================================
 * ASCII RENDERING
 * ============================================================================ */

/* Internal structure for ASCII grid */
typedef struct {
    char **grid;
    int width;
    int height;
    int label_width;
} ascii_grid_t;

static ascii_grid_t *create_ascii_grid(int width, int height) {
    ascii_grid_t *grid = malloc(sizeof(ascii_grid_t));
    if (!grid) return NULL;

    grid->width = width;
    grid->height = height;
    grid->label_width = 6;
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

static void free_ascii_grid(ascii_grid_t *grid) {
    if (grid) {
        for (int i = 0; i < grid->height; i++) {
            free(grid->grid[i]);
        }
        free(grid->grid);
        free(grid);
    }
}

static char *grid_to_string(ascii_grid_t *grid) {
    /* Calculate total size */
    size_t total = 0;
    for (int i = 0; i < grid->height; i++) {
        total += strlen(grid->grid[i]) + 1;  /* +1 for newline */
    }
    total++;  /* null terminator */

    char *result = malloc(total);
    if (!result) return NULL;

    char *ptr = result;
    for (int i = 0; i < grid->height; i++) {
        /* Trim trailing spaces */
        int len = strlen(grid->grid[i]);
        while (len > 0 && grid->grid[i][len - 1] == ' ') len--;

        memcpy(ptr, grid->grid[i], len);
        ptr += len;
        *ptr++ = '\n';
    }
    *ptr = '\0';

    return result;
}

static void draw_wire(ascii_grid_t *grid, int y, int x_start, int x_end, char wire_char) {
    for (int x = x_start; x <= x_end && x < grid->width; x++) {
        if (grid->grid[y][x] == ' ') {
            grid->grid[y][x] = wire_char;
        }
    }
}

static void draw_text(ascii_grid_t *grid, int y, int x, const char *text) {
    int len = strlen(text);
    for (int i = 0; i < len && x + i < grid->width; i++) {
        grid->grid[y][x + i] = text[i];
    }
}

static void draw_box_gate(ascii_grid_t *grid, int y, int x, const char *name, int width) {
    if (y < 1 || y >= grid->height - 1) return;

    /* Top border */
    grid->grid[y - 1][x] = '+';
    for (int i = 1; i < width - 1 && x + i < grid->width; i++) {
        grid->grid[y - 1][x + i] = '-';
    }
    grid->grid[y - 1][x + width - 1] = '+';

    /* Middle with name */
    grid->grid[y][x] = '|';
    int name_len = strlen(name);
    int name_start = x + (width - name_len) / 2;
    for (int i = 1; i < width - 1 && x + i < grid->width; i++) {
        grid->grid[y][x + i] = ' ';
    }
    draw_text(grid, y, name_start, name);
    grid->grid[y][x + width - 1] = '|';

    /* Bottom border */
    grid->grid[y + 1][x] = '+';
    for (int i = 1; i < width - 1 && x + i < grid->width; i++) {
        grid->grid[y + 1][x + i] = '-';
    }
    grid->grid[y + 1][x + width - 1] = '+';
}

static void draw_control_dot(ascii_grid_t *grid, int y, int x) {
    grid->grid[y][x] = '*';
}

static void draw_target_x(ascii_grid_t *grid, int y, int x) {
    grid->grid[y][x] = 'X';
}

static void draw_vertical_line(ascii_grid_t *grid, int y1, int y2, int x) {
    int y_min = min_int(y1, y2);
    int y_max = max_int(y1, y2);
    for (int y = y_min; y <= y_max && y < grid->height; y++) {
        if (grid->grid[y][x] == ' ' || grid->grid[y][x] == '-') {
            grid->grid[y][x] = '|';
        }
    }
}

static void draw_measurement(ascii_grid_t *grid, int y, int x, int classical_bit) {
    if (y < 1 || y >= grid->height - 1) return;

    /* Simple measurement box */
    grid->grid[y - 1][x] = '+';
    grid->grid[y - 1][x + 1] = '-';
    grid->grid[y - 1][x + 2] = '+';

    grid->grid[y][x] = '|';
    grid->grid[y][x + 1] = 'M';
    grid->grid[y][x + 2] = '|';

    grid->grid[y + 1][x] = '+';
    grid->grid[y + 1][x + 1] = '-';
    grid->grid[y + 1][x + 2] = '+';
}

static void draw_barrier_line(ascii_grid_t *grid, int y1, int y2, int x) {
    int y_min = min_int(y1, y2);
    int y_max = max_int(y1, y2);
    for (int y = y_min; y <= y_max && y < grid->height; y++) {
        grid->grid[y][x] = ':';
    }
}

char *circuit_render_ascii(const circuit_diagram_t *circuit, const render_options_t *opts) {
    if (!circuit) return NULL;

    render_options_t default_opts = circuit_default_options();
    if (!opts) opts = &default_opts;

    int depth = circuit_get_depth(circuit);
    if (depth == 0) depth = 1;

    /* Calculate dimensions */
    int label_width = 6;
    int gate_width = opts->gate_width;
    int wire_spacing = opts->wire_spacing;

    int grid_width = label_width + depth * (gate_width + 1) + 10;
    int grid_height = circuit->num_qubits * wire_spacing + 4;

    /* Add space for classical wires if needed */
    bool has_measurements = false;
    for (int i = 0; i < circuit->num_elements; i++) {
        if (circuit->elements[i].type == CIRCUIT_MEASUREMENT) {
            has_measurements = true;
            break;
        }
    }
    if (has_measurements) {
        grid_height += 3;
    }

    ascii_grid_t *grid = create_ascii_grid(grid_width, grid_height);
    if (!grid) return NULL;

    /* Draw title if present */
    int start_row = 0;
    if (circuit->title[0] != '\0') {
        draw_text(grid, 0, 0, circuit->title);
        start_row = 2;
    }

    /* Draw quantum wires and labels */
    for (int q = 0; q < circuit->num_qubits; q++) {
        int y = start_row + 1 + q * wire_spacing;

        /* Label */
        char label[16];
        snprintf(label, sizeof(label), "%s:", circuit->qubit_labels[q]);
        draw_text(grid, y, 0, label);

        /* Wire */
        draw_wire(grid, y, label_width, grid_width - 1, '-');
    }

    /* Draw classical wire if measurements exist */
    int classical_y = start_row + 1 + circuit->num_qubits * wire_spacing + 1;
    if (has_measurements) {
        draw_text(grid, classical_y, 0, "c:");
        draw_wire(grid, classical_y, label_width, grid_width - 1, '=');
    }

    /* Draw elements */
    for (int i = 0; i < circuit->num_elements; i++) {
        const circuit_element_t *elem = &circuit->elements[i];
        int x = label_width + elem->time_slot * (gate_width + 1) + 1;
        int y = start_row + 1 + elem->qubit * wire_spacing;

        switch (elem->type) {
            case CIRCUIT_GATE_SINGLE:
                draw_box_gate(grid, y, x, elem->name, gate_width);
                break;

            case CIRCUIT_GATE_CONTROLLED: {
                int target_y = start_row + 1 + elem->target_qubit * wire_spacing;
                int x_center = x + gate_width / 2;

                /* Control dot */
                draw_control_dot(grid, y, x_center);

                /* Vertical line */
                draw_vertical_line(grid, y, target_y, x_center);

                /* Target */
                if (strcmp(elem->name, "X") == 0) {
                    draw_target_x(grid, target_y, x_center);
                } else if (strcmp(elem->name, "Z") == 0) {
                    draw_control_dot(grid, target_y, x_center);  /* CZ has two dots */
                } else {
                    draw_box_gate(grid, target_y, x, elem->name, gate_width);
                }
                break;
            }

            case CIRCUIT_GATE_MULTI: {
                int target_y = start_row + 1 + elem->target_qubit * wire_spacing;
                int x_center = x + gate_width / 2;

                if (strcmp(elem->name, "SWAP") == 0) {
                    /* SWAP: X marks on both qubits */
                    draw_text(grid, y, x_center, "X");
                    draw_text(grid, target_y, x_center, "X");
                    draw_vertical_line(grid, y, target_y, x_center);
                } else if (strcmp(elem->name, "CCX") == 0) {
                    /* Toffoli: two control dots, one target */
                    int ctrl2_y = start_row + 1 + elem->control_qubit2 * wire_spacing;
                    draw_control_dot(grid, y, x_center);
                    draw_control_dot(grid, ctrl2_y, x_center);
                    draw_target_x(grid, target_y, x_center);
                    draw_vertical_line(grid, min_int(y, min_int(ctrl2_y, target_y)),
                                      max_int(y, max_int(ctrl2_y, target_y)), x_center);
                } else if (strcmp(elem->name, "CSWAP") == 0) {
                    /* Fredkin: one control, two swap targets */
                    int swap2_y = start_row + 1 + elem->control_qubit2 * wire_spacing;
                    draw_control_dot(grid, y, x_center);
                    draw_text(grid, target_y, x_center, "X");
                    draw_text(grid, swap2_y, x_center, "X");
                    draw_vertical_line(grid, min_int(y, min_int(swap2_y, target_y)),
                                      max_int(y, max_int(swap2_y, target_y)), x_center);
                }
                break;
            }

            case CIRCUIT_MEASUREMENT:
                draw_measurement(grid, y, x, elem->classical_bit);
                /* Draw connection to classical wire */
                if (has_measurements) {
                    int x_center = x + 1;
                    for (int cy = y + 2; cy < classical_y; cy++) {
                        if (grid->grid[cy][x_center] == ' ' || grid->grid[cy][x_center] == '-') {
                            grid->grid[cy][x_center] = '|';
                        }
                    }
                }
                break;

            case CIRCUIT_BARRIER:
                if (opts->show_barriers) {
                    int min_y = grid_height;
                    int max_y = 0;
                    for (int j = 0; j < elem->num_barrier_qubits; j++) {
                        int qy = start_row + 1 + elem->barrier_qubits[j] * wire_spacing;
                        if (qy < min_y) min_y = qy;
                        if (qy > max_y) max_y = qy;
                    }
                    int x_center = x + gate_width / 2;
                    draw_barrier_line(grid, min_y - 1, max_y + 1, x_center);
                }
                break;

            case CIRCUIT_RESET:
                draw_box_gate(grid, y, x, "|0>", gate_width);
                break;

            case CIRCUIT_LABEL:
                draw_text(grid, y, x, elem->name);
                break;
        }
    }

    char *result = grid_to_string(grid);
    free_ascii_grid(grid);
    return result;
}

void circuit_print_ascii(const circuit_diagram_t *circuit) {
    char *ascii = circuit_render_ascii(circuit, NULL);
    if (ascii) {
        printf("%s", ascii);
        free(ascii);
    }
}

void circuit_print_ascii_opts(const circuit_diagram_t *circuit, const render_options_t *opts) {
    char *ascii = circuit_render_ascii(circuit, opts);
    if (ascii) {
        printf("%s", ascii);
        free(ascii);
    }
}

/* ============================================================================
 * SVG RENDERING
 * ============================================================================ */

/* String builder for SVG generation */
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

#include <stdarg.h>

char *circuit_render_svg(const circuit_diagram_t *circuit, const render_options_t *opts) {
    if (!circuit) return NULL;

    render_options_t default_opts = circuit_default_options();
    if (!opts) opts = &default_opts;

    int depth = circuit_get_depth(circuit);
    if (depth == 0) depth = 1;

    /* Calculate dimensions */
    int gate_width = 40;
    int gate_height = 30;
    int wire_spacing = 50;
    int label_width = 60;
    int padding = 20;

    int width = label_width + depth * (gate_width + 20) + padding * 2 + 50;
    int height = circuit->num_qubits * wire_spacing + padding * 2 + 40;

    string_builder_t *sb = sb_create();
    if (!sb) return NULL;

    /* SVG header */
    sb_appendf(sb, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    sb_appendf(sb, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
               width, height, width, height);

    /* Styles */
    sb_append(sb, "<defs>\n");
    sb_append(sb, "  <style>\n");
    sb_append(sb, "    .gate { fill: white; stroke: #333; stroke-width: 1.5; }\n");
    sb_append(sb, "    .gate-text { font-family: 'Courier New', monospace; font-size: 14px; text-anchor: middle; dominant-baseline: middle; }\n");
    sb_append(sb, "    .wire { stroke: #333; stroke-width: 1.5; fill: none; }\n");
    sb_append(sb, "    .control { fill: #333; }\n");
    sb_append(sb, "    .target { fill: white; stroke: #333; stroke-width: 1.5; }\n");
    sb_append(sb, "    .measure { fill: #f0f0f0; stroke: #333; stroke-width: 1.5; }\n");
    sb_append(sb, "    .label { font-family: 'Courier New', monospace; font-size: 12px; text-anchor: end; dominant-baseline: middle; }\n");
    sb_append(sb, "    .title { font-family: 'Courier New', monospace; font-size: 16px; font-weight: bold; text-anchor: middle; }\n");
    sb_append(sb, "    .barrier { stroke: #999; stroke-width: 1; stroke-dasharray: 4,4; }\n");
    sb_append(sb, "  </style>\n");
    sb_append(sb, "</defs>\n");

    /* Background */
    sb_appendf(sb, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", width, height);

    /* Title */
    int title_y = padding;
    if (circuit->title[0] != '\0') {
        sb_appendf(sb, "<text x=\"%d\" y=\"%d\" class=\"title\">%s</text>\n",
                   width / 2, title_y, circuit->title);
        title_y += 25;
    }

    /* Draw wires and labels */
    for (int q = 0; q < circuit->num_qubits; q++) {
        int y = title_y + padding + q * wire_spacing + wire_spacing / 2;

        /* Label */
        sb_appendf(sb, "<text x=\"%d\" y=\"%d\" class=\"label\">%s</text>\n",
                   label_width - 5, y, circuit->qubit_labels[q]);

        /* Wire */
        sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                   label_width, y, width - padding, y);
    }

    /* Draw elements */
    for (int i = 0; i < circuit->num_elements; i++) {
        const circuit_element_t *elem = &circuit->elements[i];
        int x = label_width + padding + elem->time_slot * (gate_width + 20);
        int y = title_y + padding + elem->qubit * wire_spacing + wire_spacing / 2;

        switch (elem->type) {
            case CIRCUIT_GATE_SINGLE: {
                /* Gate box */
                sb_appendf(sb, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"3\" class=\"gate\"/>\n",
                           x, y - gate_height / 2, gate_width, gate_height);
                sb_appendf(sb, "<text x=\"%d\" y=\"%d\" class=\"gate-text\">%s</text>\n",
                           x + gate_width / 2, y, elem->name);
                break;
            }

            case CIRCUIT_GATE_CONTROLLED: {
                int target_y = title_y + padding + elem->target_qubit * wire_spacing + wire_spacing / 2;
                int x_center = x + gate_width / 2;

                /* Vertical line */
                sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                           x_center, y, x_center, target_y);

                /* Control dot */
                sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"5\" class=\"control\"/>\n",
                           x_center, y);

                /* Target */
                if (strcmp(elem->name, "X") == 0) {
                    /* XOR symbol for CNOT */
                    sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"12\" class=\"target\"/>\n",
                               x_center, target_y);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center - 12, target_y, x_center + 12, target_y);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center, target_y - 12, x_center, target_y + 12);
                } else if (strcmp(elem->name, "Z") == 0) {
                    /* CZ: second control dot */
                    sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"5\" class=\"control\"/>\n",
                               x_center, target_y);
                } else {
                    /* Other controlled gates: box on target */
                    sb_appendf(sb, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"3\" class=\"gate\"/>\n",
                               x, target_y - gate_height / 2, gate_width, gate_height);
                    sb_appendf(sb, "<text x=\"%d\" y=\"%d\" class=\"gate-text\">%s</text>\n",
                               x + gate_width / 2, target_y, elem->name);
                }
                break;
            }

            case CIRCUIT_GATE_MULTI: {
                int target_y = title_y + padding + elem->target_qubit * wire_spacing + wire_spacing / 2;
                int x_center = x + gate_width / 2;

                if (strcmp(elem->name, "SWAP") == 0) {
                    /* SWAP: X marks connected by line */
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center, y, x_center, target_y);
                    /* X on first qubit */
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center - 8, y - 8, x_center + 8, y + 8);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center + 8, y - 8, x_center - 8, y + 8);
                    /* X on second qubit */
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center - 8, target_y - 8, x_center + 8, target_y + 8);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center + 8, target_y - 8, x_center - 8, target_y + 8);
                } else if (strcmp(elem->name, "CCX") == 0) {
                    int ctrl2_y = title_y + padding + elem->control_qubit2 * wire_spacing + wire_spacing / 2;
                    int min_y = min_int(y, min_int(ctrl2_y, target_y));
                    int max_y = max_int(y, max_int(ctrl2_y, target_y));

                    /* Vertical line through all qubits */
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center, min_y, x_center, max_y);

                    /* Two control dots */
                    sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"5\" class=\"control\"/>\n",
                               x_center, y);
                    sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"5\" class=\"control\"/>\n",
                               x_center, ctrl2_y);

                    /* Target XOR */
                    sb_appendf(sb, "<circle cx=\"%d\" cy=\"%d\" r=\"12\" class=\"target\"/>\n",
                               x_center, target_y);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center - 12, target_y, x_center + 12, target_y);
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                               x_center, target_y - 12, x_center, target_y + 12);
                }
                break;
            }

            case CIRCUIT_MEASUREMENT: {
                /* Measurement box */
                sb_appendf(sb, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"3\" class=\"measure\"/>\n",
                           x, y - gate_height / 2, gate_width, gate_height);
                /* Meter arc */
                sb_appendf(sb, "<path d=\"M %d %d A 8 8 0 0 1 %d %d\" class=\"wire\"/>\n",
                           x + gate_width / 2 - 8, y + 2, x + gate_width / 2 + 8, y + 2);
                /* Meter arrow */
                sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"wire\"/>\n",
                           x + gate_width / 2, y + 2, x + gate_width / 2 + 6, y - 8);
                break;
            }

            case CIRCUIT_BARRIER: {
                if (opts->show_barriers) {
                    int min_y = height;
                    int max_y = 0;
                    for (int j = 0; j < elem->num_barrier_qubits; j++) {
                        int qy = title_y + padding + elem->barrier_qubits[j] * wire_spacing + wire_spacing / 2;
                        if (qy < min_y) min_y = qy;
                        if (qy > max_y) max_y = qy;
                    }
                    sb_appendf(sb, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"barrier\"/>\n",
                               x + gate_width / 2, min_y - 15, x + gate_width / 2, max_y + 15);
                }
                break;
            }

            case CIRCUIT_RESET: {
                sb_appendf(sb, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" rx=\"3\" class=\"gate\"/>\n",
                           x, y - gate_height / 2, gate_width, gate_height);
                sb_appendf(sb, "<text x=\"%d\" y=\"%d\" class=\"gate-text\">|0&gt;</text>\n",
                           x + gate_width / 2, y);
                break;
            }

            default:
                break;
        }
    }

    sb_append(sb, "</svg>\n");

    char *result = sb->buffer;
    sb->buffer = NULL;
    sb_free(sb);
    return result;
}

int circuit_save_svg(const circuit_diagram_t *circuit, const char *filename,
                     const render_options_t *opts) {
    char *svg = circuit_render_svg(circuit, opts);
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

char *circuit_render_latex(const circuit_diagram_t *circuit) {
    if (!circuit) return NULL;

    int depth = circuit_get_depth(circuit);

    string_builder_t *sb = sb_create();
    if (!sb) return NULL;

    sb_append(sb, "% Requires: \\usepackage{quantikz}\n");
    sb_append(sb, "\\begin{quantikz}\n");

    for (int q = 0; q < circuit->num_qubits; q++) {
        /* Qubit label */
        sb_appendf(sb, "\\lstick{$%s$}", circuit->qubit_labels[q]);

        /* Gates at each time slot */
        for (int t = 0; t <= depth; t++) {
            const circuit_element_t *elem = circuit_get_element_at(circuit, q, t);

            if (!elem) {
                sb_append(sb, " & \\qw");
            } else if (elem->type == CIRCUIT_BARRIER) {
                sb_append(sb, " & \\qw");  /* quantikz handles barriers differently */
            } else if (elem->qubit == q) {
                /* This qubit is the primary qubit for this element */
                switch (elem->type) {
                    case CIRCUIT_GATE_SINGLE:
                        if (elem->angle != 0) {
                            char angle_str[16];
                            circuit_format_angle(elem->angle, angle_str, sizeof(angle_str));
                            sb_appendf(sb, " & \\gate{%s(%s)}", elem->name, angle_str);
                        } else {
                            sb_appendf(sb, " & \\gate{%s}", elem->name);
                        }
                        break;

                    case CIRCUIT_GATE_CONTROLLED: {
                        int delta = elem->target_qubit - elem->qubit;
                        sb_appendf(sb, " & \\ctrl{%d}", delta);
                        break;
                    }

                    case CIRCUIT_GATE_MULTI:
                        if (strcmp(elem->name, "SWAP") == 0) {
                            int delta = elem->target_qubit - elem->qubit;
                            sb_appendf(sb, " & \\swap{%d}", delta);
                        } else if (strcmp(elem->name, "CCX") == 0) {
                            int delta = elem->target_qubit - elem->qubit;
                            sb_appendf(sb, " & \\ctrl{%d}", delta);
                        } else {
                            sb_append(sb, " & \\qw");
                        }
                        break;

                    case CIRCUIT_MEASUREMENT:
                        sb_append(sb, " & \\meter{}");
                        break;

                    case CIRCUIT_RESET:
                        sb_append(sb, " & \\gate{|0\\rangle}");
                        break;

                    default:
                        sb_append(sb, " & \\qw");
                }
            } else if (elem->target_qubit == q) {
                /* This is the target of a controlled gate */
                if (elem->type == CIRCUIT_GATE_CONTROLLED) {
                    if (strcmp(elem->name, "X") == 0) {
                        sb_append(sb, " & \\targ{}");
                    } else if (strcmp(elem->name, "Z") == 0) {
                        sb_append(sb, " & \\control{}");
                    } else {
                        sb_appendf(sb, " & \\gate{%s}", elem->name);
                    }
                } else if (elem->type == CIRCUIT_GATE_MULTI) {
                    if (strcmp(elem->name, "SWAP") == 0) {
                        sb_append(sb, " & \\targX{}");
                    } else if (strcmp(elem->name, "CCX") == 0) {
                        sb_append(sb, " & \\targ{}");
                    } else {
                        sb_append(sb, " & \\qw");
                    }
                } else {
                    sb_append(sb, " & \\qw");
                }
            } else if (elem->control_qubit2 == q) {
                /* This is the second control of a Toffoli */
                if (strcmp(elem->name, "CCX") == 0) {
                    int delta = elem->target_qubit - q;
                    sb_appendf(sb, " & \\ctrl{%d}", delta);
                } else {
                    sb_append(sb, " & \\qw");
                }
            } else {
                sb_append(sb, " & \\qw");
            }
        }

        /* End of row */
        if (q < circuit->num_qubits - 1) {
            sb_append(sb, " \\\\\n");
        } else {
            sb_append(sb, "\n");
        }
    }

    sb_append(sb, "\\end{quantikz}");

    char *result = sb->buffer;
    sb->buffer = NULL;
    sb_free(sb);
    return result;
}

int circuit_save_latex(const circuit_diagram_t *circuit, const char *filename) {
    char *latex = circuit_render_latex(circuit);
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

/* ============================================================================
 * CIRCUIT FROM STRING
 * ============================================================================ */

circuit_diagram_t *circuit_from_string(int num_qubits, const char *gate_string) {
    if (!gate_string || num_qubits < 1) return NULL;

    circuit_diagram_t *circuit = circuit_create(num_qubits);
    if (!circuit) return NULL;

    char *str = strdup(gate_string);
    if (!str) {
        circuit_free(circuit);
        return NULL;
    }

    char *token = strtok(str, ";");
    while (token) {
        /* Skip leading whitespace */
        while (*token == ' ') token++;

        char gate[32];
        int q1 = -1, q2 = -1, q3 = -1;
        double angle = 0;

        /* Parse gate and arguments */
        if (sscanf(token, "%31s %d %d %d", gate, &q1, &q2, &q3) >= 2) {
            /* Gate with at least one qubit argument */
            if (strcmp(gate, "H") == 0 || strcmp(gate, "X") == 0 ||
                strcmp(gate, "Y") == 0 || strcmp(gate, "Z") == 0 ||
                strcmp(gate, "S") == 0 || strcmp(gate, "T") == 0) {
                circuit_add_gate(circuit, q1, gate);
            } else if (strcmp(gate, "CNOT") == 0 || strcmp(gate, "CX") == 0) {
                circuit_add_controlled(circuit, q1, q2, "X");
            } else if (strcmp(gate, "CZ") == 0) {
                circuit_add_controlled(circuit, q1, q2, "Z");
            } else if (strcmp(gate, "SWAP") == 0) {
                circuit_add_swap(circuit, q1, q2);
            } else if (strcmp(gate, "CCX") == 0 || strcmp(gate, "TOFFOLI") == 0) {
                circuit_add_toffoli(circuit, q1, q2, q3);
            } else if (strcmp(gate, "M") == 0) {
                circuit_add_measurement(circuit, q1, q1);
            } else if (strcmp(gate, "BARRIER") == 0) {
                circuit_add_barrier_all(circuit);
            } else if (strncmp(gate, "RX", 2) == 0 || strncmp(gate, "RY", 2) == 0 ||
                       strncmp(gate, "RZ", 2) == 0) {
                /* Parse angle from gate name like "RX(pi/2)" */
                char *angle_start = strchr(gate, '(');
                if (angle_start) {
                    *angle_start = '\0';
                    angle_start++;
                    char *angle_end = strchr(angle_start, ')');
                    if (angle_end) *angle_end = '\0';

                    if (strstr(angle_start, "pi")) {
                        /* Parse pi expression */
                        if (strcmp(angle_start, "pi") == 0) {
                            angle = PI;
                        } else if (strcmp(angle_start, "pi/2") == 0) {
                            angle = PI / 2;
                        } else if (strcmp(angle_start, "pi/4") == 0) {
                            angle = PI / 4;
                        } else if (strcmp(angle_start, "-pi/2") == 0) {
                            angle = -PI / 2;
                        } else {
                            angle = atof(angle_start) * PI;
                        }
                    } else {
                        angle = atof(angle_start);
                    }
                }
                circuit_add_rotation(circuit, q1, gate, angle);
            }
        }

        token = strtok(NULL, ";");
    }

    free(str);
    return circuit;
}
