/**
 * @file circuit_diagram.h
 * @brief Quantum Circuit Diagram Renderer
 *
 * Generate publication-quality quantum circuit diagrams in multiple formats:
 * - ASCII (terminal output)
 * - SVG (web/publication)
 * - LaTeX/TikZ (papers using quantikz package)
 *
 * CIRCUIT ELEMENTS:
 * =================
 * - Single-qubit gates: H, X, Y, Z, S, T, Rx, Ry, Rz
 * - Controlled gates: CNOT, CZ, CRx, CRy, CRz
 * - Multi-qubit gates: SWAP, Toffoli, Fredkin
 * - Measurements: Standard and mid-circuit
 * - Barriers: Visual separators
 * - Classical wires: For measurement results
 *
 * ASCII RENDERING:
 * ================
 *      +---------+          +---------+     +---+
 * q0: -|    H    |----*-----|    H    |--*--|M_0|---
 *      +---------+    |     +---------+  |  +---+
 *                     |  +---------+     |    ||
 * q1: ----------------+--|    X    |-----+----||----
 *                        +---------+     |    ||
 *      +---------+  +---+               +---+ || +---+
 * q2: -|    H    |--| X |---------------| X |--|-|M_1|
 *      +---------+  +---+               +---+  | +---+
 *                                              ||  ||
 * c:  =========================================++==++==
 *                                              0   1
 *
 * USAGE:
 * ======
 *     circuit_diagram_t *circuit = circuit_create(3);
 *     circuit_set_title(circuit, "Bell State Preparation");
 *     circuit_add_gate(circuit, 0, "H");
 *     circuit_add_controlled(circuit, 0, 1, "X");
 *     circuit_add_measurement(circuit, 0, 0);
 *     circuit_add_measurement(circuit, 1, 1);
 *     circuit_print_ascii(circuit);
 *     circuit_save_svg(circuit, "bell.svg", NULL);
 *     circuit_free(circuit);
 */

#ifndef CIRCUIT_DIAGRAM_H
#define CIRCUIT_DIAGRAM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define CIRCUIT_MAX_QUBITS 32
#define CIRCUIT_MAX_CLASSICAL 32
#define CIRCUIT_MAX_ELEMENTS 1024
#define CIRCUIT_MAX_LABEL_LEN 32
#define CIRCUIT_MAX_TITLE_LEN 128

/* ============================================================================
 * TYPES
 * ============================================================================ */

/**
 * Circuit element types
 */
typedef enum {
    CIRCUIT_GATE_SINGLE,      /**< Single-qubit gate (H, X, Y, Z, S, T, Rx, Ry, Rz) */
    CIRCUIT_GATE_CONTROLLED,  /**< Controlled gate (CNOT, CZ, CRx, etc.) */
    CIRCUIT_GATE_MULTI,       /**< Multi-qubit gate (SWAP, Toffoli, Fredkin) */
    CIRCUIT_MEASUREMENT,      /**< Measurement operation */
    CIRCUIT_BARRIER,          /**< Visual barrier/separator */
    CIRCUIT_RESET,            /**< Qubit reset to |0> */
    CIRCUIT_LABEL             /**< Custom label/annotation */
} circuit_element_type_t;

/**
 * Single circuit element
 */
typedef struct {
    circuit_element_type_t type;
    int qubit;                /**< Primary qubit */
    int target_qubit;         /**< Target for controlled gates */
    int control_qubit2;       /**< Second control for Toffoli (CCX) */
    char name[CIRCUIT_MAX_LABEL_LEN];  /**< Gate name ("H", "CNOT", "Rx(pi/4)") */
    double angle;             /**< Rotation angle for Rx, Ry, Rz gates */
    int time_slot;            /**< Horizontal position (column) */
    int classical_bit;        /**< Classical bit for measurement */
    uint32_t color;           /**< RGBA color for rendering (0 = default) */
    int barrier_qubits[CIRCUIT_MAX_QUBITS];  /**< Qubits covered by barrier */
    int num_barrier_qubits;   /**< Number of qubits in barrier */
} circuit_element_t;

/**
 * Complete circuit representation
 */
typedef struct {
    int num_qubits;
    int num_classical;        /**< Number of classical bits for measurement */
    int num_elements;
    int capacity;
    circuit_element_t *elements;
    char title[CIRCUIT_MAX_TITLE_LEN];
    char qubit_labels[CIRCUIT_MAX_QUBITS][CIRCUIT_MAX_LABEL_LEN];
    int current_time[CIRCUIT_MAX_QUBITS];  /**< Current time slot per qubit */
} circuit_diagram_t;

/**
 * Rendering options for output customization
 */
typedef struct {
    int gate_width;           /**< Character/pixel width of gates */
    int gate_height;          /**< Height of gates */
    int wire_spacing;         /**< Vertical spacing between qubits */
    bool show_grid;           /**< Show time grid lines */
    bool show_labels;         /**< Show qubit labels */
    bool show_barriers;       /**< Show barrier lines */
    bool color_phases;        /**< Color gates by phase rotation */
    bool compact;             /**< Use compact gate names */
    char font_family[64];     /**< Font for SVG/PDF output */
    int font_size;            /**< Font size in points */
    int svg_width;            /**< SVG canvas width */
    int svg_height;           /**< SVG canvas height */
} render_options_t;

/* ============================================================================
 * CIRCUIT CREATION AND MANAGEMENT
 * ============================================================================ */

/**
 * Create a new circuit diagram
 *
 * @param num_qubits Number of qubits in the circuit
 * @return Pointer to new circuit, or NULL on failure
 */
circuit_diagram_t *circuit_create(int num_qubits);

/**
 * Create circuit with both quantum and classical registers
 *
 * @param num_qubits Number of quantum bits
 * @param num_classical Number of classical bits
 * @return Pointer to new circuit, or NULL on failure
 */
circuit_diagram_t *circuit_create_with_classical(int num_qubits, int num_classical);

/**
 * Free circuit diagram and all resources
 *
 * @param circuit Circuit to free
 */
void circuit_free(circuit_diagram_t *circuit);

/**
 * Set circuit title
 *
 * @param circuit Circuit to modify
 * @param title Title string
 */
void circuit_set_title(circuit_diagram_t *circuit, const char *title);

/**
 * Set custom label for a qubit wire
 *
 * @param circuit Circuit to modify
 * @param qubit Qubit index
 * @param label Label string (e.g., "|psi>", "|0>", "ancilla")
 */
void circuit_set_qubit_label(circuit_diagram_t *circuit, int qubit, const char *label);

/**
 * Get circuit depth (number of time steps)
 *
 * @param circuit Circuit to query
 * @return Maximum time slot used
 */
int circuit_get_depth(const circuit_diagram_t *circuit);

/**
 * Clear all elements from circuit
 *
 * @param circuit Circuit to clear
 */
void circuit_clear(circuit_diagram_t *circuit);

/* ============================================================================
 * ADDING CIRCUIT ELEMENTS
 * ============================================================================ */

/**
 * Add single-qubit gate
 *
 * @param circuit Circuit to modify
 * @param qubit Target qubit
 * @param name Gate name ("H", "X", "Y", "Z", "S", "T")
 * @return Time slot where gate was placed
 */
int circuit_add_gate(circuit_diagram_t *circuit, int qubit, const char *name);

/**
 * Add rotation gate with angle
 *
 * @param circuit Circuit to modify
 * @param qubit Target qubit
 * @param name Gate name ("Rx", "Ry", "Rz")
 * @param angle Rotation angle in radians
 * @return Time slot where gate was placed
 */
int circuit_add_rotation(circuit_diagram_t *circuit, int qubit, const char *name, double angle);

/**
 * Add controlled gate (CNOT, CZ, etc.)
 *
 * @param circuit Circuit to modify
 * @param control Control qubit
 * @param target Target qubit
 * @param name Gate name ("X" for CNOT, "Z" for CZ, etc.)
 * @return Time slot where gate was placed
 */
int circuit_add_controlled(circuit_diagram_t *circuit, int control, int target, const char *name);

/**
 * Add controlled rotation gate
 *
 * @param circuit Circuit to modify
 * @param control Control qubit
 * @param target Target qubit
 * @param name Gate name ("Rx", "Ry", "Rz", "P")
 * @param angle Rotation angle in radians
 * @return Time slot where gate was placed
 */
int circuit_add_controlled_rotation(circuit_diagram_t *circuit, int control, int target,
                                    const char *name, double angle);

/**
 * Add Toffoli (CCX) gate
 *
 * @param circuit Circuit to modify
 * @param ctrl1 First control qubit
 * @param ctrl2 Second control qubit
 * @param target Target qubit
 * @return Time slot where gate was placed
 */
int circuit_add_toffoli(circuit_diagram_t *circuit, int ctrl1, int ctrl2, int target);

/**
 * Add Fredkin (CSWAP) gate
 *
 * @param circuit Circuit to modify
 * @param control Control qubit
 * @param target1 First target qubit
 * @param target2 Second target qubit
 * @return Time slot where gate was placed
 */
int circuit_add_fredkin(circuit_diagram_t *circuit, int control, int target1, int target2);

/**
 * Add SWAP gate
 *
 * @param circuit Circuit to modify
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return Time slot where gate was placed
 */
int circuit_add_swap(circuit_diagram_t *circuit, int qubit1, int qubit2);

/**
 * Add measurement to classical bit
 *
 * @param circuit Circuit to modify
 * @param qubit Qubit to measure
 * @param classical_bit Classical bit to store result
 * @return Time slot where measurement was placed
 */
int circuit_add_measurement(circuit_diagram_t *circuit, int qubit, int classical_bit);

/**
 * Add visual barrier across specified qubits
 *
 * @param circuit Circuit to modify
 * @param qubits Array of qubit indices
 * @param num_qubits Number of qubits in barrier
 * @return Time slot where barrier was placed
 */
int circuit_add_barrier(circuit_diagram_t *circuit, const int *qubits, int num_qubits);

/**
 * Add barrier across all qubits
 *
 * @param circuit Circuit to modify
 * @return Time slot where barrier was placed
 */
int circuit_add_barrier_all(circuit_diagram_t *circuit);

/**
 * Add qubit reset to |0>
 *
 * @param circuit Circuit to modify
 * @param qubit Qubit to reset
 * @return Time slot where reset was placed
 */
int circuit_add_reset(circuit_diagram_t *circuit, int qubit);

/* ============================================================================
 * RENDERING OPTIONS
 * ============================================================================ */

/**
 * Get default rendering options
 *
 * @return Default options structure
 */
render_options_t circuit_default_options(void);

/**
 * Get compact rendering options (smaller gates)
 *
 * @return Compact options structure
 */
render_options_t circuit_compact_options(void);

/**
 * Get publication-quality rendering options
 *
 * @return Publication options structure
 */
render_options_t circuit_publication_options(void);

/* ============================================================================
 * ASCII RENDERING
 * ============================================================================ */

/**
 * Render circuit to ASCII string
 *
 * @param circuit Circuit to render
 * @param opts Rendering options (NULL for defaults)
 * @return Allocated string with ASCII diagram (caller must free)
 */
char *circuit_render_ascii(const circuit_diagram_t *circuit, const render_options_t *opts);

/**
 * Print circuit to stdout
 *
 * @param circuit Circuit to print
 */
void circuit_print_ascii(const circuit_diagram_t *circuit);

/**
 * Print circuit with custom options
 *
 * @param circuit Circuit to print
 * @param opts Rendering options
 */
void circuit_print_ascii_opts(const circuit_diagram_t *circuit, const render_options_t *opts);

/* ============================================================================
 * SVG RENDERING
 * ============================================================================ */

/**
 * Render circuit to SVG string
 *
 * @param circuit Circuit to render
 * @param opts Rendering options (NULL for defaults)
 * @return Allocated string with SVG content (caller must free)
 */
char *circuit_render_svg(const circuit_diagram_t *circuit, const render_options_t *opts);

/**
 * Save circuit to SVG file
 *
 * @param circuit Circuit to save
 * @param filename Output filename
 * @param opts Rendering options (NULL for defaults)
 * @return 0 on success, -1 on error
 */
int circuit_save_svg(const circuit_diagram_t *circuit, const char *filename,
                     const render_options_t *opts);

/* ============================================================================
 * LATEX RENDERING
 * ============================================================================ */

/**
 * Render circuit to LaTeX/quantikz format
 *
 * @param circuit Circuit to render
 * @return Allocated string with LaTeX code (caller must free)
 */
char *circuit_render_latex(const circuit_diagram_t *circuit);

/**
 * Save circuit to LaTeX file
 *
 * @param circuit Circuit to save
 * @param filename Output filename
 * @return 0 on success, -1 on error
 */
int circuit_save_latex(const circuit_diagram_t *circuit, const char *filename);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Get element at specific qubit and time
 *
 * @param circuit Circuit to query
 * @param qubit Qubit index
 * @param time Time slot
 * @return Pointer to element, or NULL if none
 */
const circuit_element_t *circuit_get_element_at(const circuit_diagram_t *circuit,
                                                int qubit, int time);

/**
 * Format angle as string (pi fractions when possible)
 *
 * @param angle Angle in radians
 * @param buffer Output buffer
 * @param buffer_size Size of buffer
 * @return Pointer to buffer
 */
char *circuit_format_angle(double angle, char *buffer, size_t buffer_size);

/**
 * Create circuit from gate sequence string
 *
 * Example: "H 0; CNOT 0 1; M 0; M 1"
 *
 * @param num_qubits Number of qubits
 * @param gate_string String describing gates
 * @return New circuit, or NULL on error
 */
circuit_diagram_t *circuit_from_string(int num_qubits, const char *gate_string);

#ifdef __cplusplus
}
#endif

#endif /* CIRCUIT_DIAGRAM_H */
