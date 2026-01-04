/**
 * @file feynman_diagram.h
 * @brief Feynman Diagram Renderer for Quantum Field Theory Visualization
 *
 * Generate publication-quality Feynman diagrams in multiple formats:
 * - ASCII (terminal output)
 * - SVG (web/publication)
 * - LaTeX/TikZ-Feynman (papers)
 *
 * PARTICLE TYPES:
 * ===============
 * | Particle        | Line Style      | Physics                    |
 * |-----------------|-----------------|----------------------------|
 * | Fermion         | -> solid arrow  | Electron, quark, neutrino  |
 * | Antifermion     | <- solid arrow  | Positron, antiquark        |
 * | Photon          | ~~~~ wavy       | Electromagnetic force      |
 * | Gluon           | @@@@ curly      | Strong force               |
 * | W/Z Boson       | ---- dashed     | Weak force                 |
 * | Higgs           | .... dotted     | Higgs mechanism            |
 * | Scalar          | ---- dashed     | Generic scalar field       |
 *
 * STANDARD PROCESSES:
 * ===================
 * - QED vertex: e- -> e- + photon
 * - Pair annihilation: e+ e- -> photon photon
 * - Compton scattering: e- + photon -> e- + photon
 * - Electron self-energy (1-loop)
 * - Vacuum polarization (1-loop)
 *
 * USAGE:
 * ======
 *     feynman_diagram_t *fd = feynman_create("e+ e- -> mu+ mu-");
 *
 *     int v1 = feynman_add_vertex(fd, -2, 1);
 *     int v2 = feynman_add_vertex(fd, -2, -1);
 *     int v3 = feynman_add_vertex(fd, 0, 0);
 *     int v4 = feynman_add_vertex(fd, 2, 1);
 *     int v5 = feynman_add_vertex(fd, 2, -1);
 *
 *     feynman_add_fermion(fd, v1, v3, "e-");
 *     feynman_add_antifermion(fd, v2, v3, "e+");
 *     feynman_add_photon(fd, v3, v3, "gamma");
 *     feynman_add_fermion(fd, v3, v4, "mu-");
 *     feynman_add_antifermion(fd, v3, v5, "mu+");
 *
 *     feynman_print_ascii(fd);
 *     feynman_save_svg(fd, "ee_mumu.svg", 500, 400);
 *     feynman_free(fd);
 */

#ifndef FEYNMAN_DIAGRAM_H
#define FEYNMAN_DIAGRAM_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define FEYNMAN_MAX_VERTICES 64
#define FEYNMAN_MAX_PROPAGATORS 128
#define FEYNMAN_MAX_LABEL_LEN 32
#define FEYNMAN_MAX_TITLE_LEN 128

/* ============================================================================
 * TYPES
 * ============================================================================ */

/**
 * Particle types for propagators
 */
typedef enum {
    PARTICLE_FERMION,         /**< Electron, muon, quark, neutrino */
    PARTICLE_ANTIFERMION,     /**< Positron, antimuon, antiquark */
    PARTICLE_PHOTON,          /**< Electromagnetic force carrier (gamma) */
    PARTICLE_GLUON,           /**< Strong force carrier (g) */
    PARTICLE_W_BOSON,         /**< Weak force carrier (W+/W-) */
    PARTICLE_Z_BOSON,         /**< Weak force carrier (Z0) */
    PARTICLE_HIGGS,           /**< Higgs boson (H) */
    PARTICLE_SCALAR,          /**< Generic scalar field */
    PARTICLE_GHOST,           /**< Faddeev-Popov ghost */
    PARTICLE_GRAVITON         /**< Graviton (theoretical) */
} particle_type_t;

/**
 * Vertex (interaction point)
 */
typedef struct {
    double x, y;              /**< Position in diagram coordinates */
    int id;                   /**< Unique vertex ID */
    char label[FEYNMAN_MAX_LABEL_LEN];  /**< Optional label (e.g., "g_s") */
    bool is_external;         /**< True for external particle attachment */
} feynman_vertex_t;

/**
 * Propagator (line between vertices)
 */
typedef struct {
    int from_vertex;          /**< Starting vertex ID */
    int to_vertex;            /**< Ending vertex ID */
    particle_type_t type;     /**< Particle type determines line style */
    char label[FEYNMAN_MAX_LABEL_LEN];  /**< Particle label (e-, gamma, etc.) */
    bool arrow_forward;       /**< Arrow direction for fermions */
    bool is_external;         /**< External (incoming/outgoing) line */
    double momentum[4];       /**< Optional 4-momentum (p0, p1, p2, p3) */
    bool show_momentum;       /**< Whether to display momentum label */
} feynman_propagator_t;

/**
 * Complete Feynman diagram
 */
typedef struct {
    feynman_vertex_t vertices[FEYNMAN_MAX_VERTICES];
    int num_vertices;
    feynman_propagator_t propagators[FEYNMAN_MAX_PROPAGATORS];
    int num_propagators;
    char title[FEYNMAN_MAX_TITLE_LEN];
    char process[FEYNMAN_MAX_TITLE_LEN];  /**< Process notation (e.g., "e+ e- -> mu+ mu-") */
    int loop_order;           /**< 0 = tree, 1 = one-loop, etc. */
    double min_x, max_x;      /**< Bounding box */
    double min_y, max_y;
} feynman_diagram_t;

/**
 * Rendering options
 */
typedef struct {
    int width;                /**< Output width (pixels for SVG, chars for ASCII) */
    int height;               /**< Output height */
    bool show_labels;         /**< Show particle labels */
    bool show_momentum;       /**< Show momentum arrows/labels */
    bool show_vertices;       /**< Show vertex dots */
    int wave_amplitude;       /**< Amplitude of wavy/curly lines */
    int wave_frequency;       /**< Frequency of wavy/curly lines */
    char font_family[64];     /**< Font for labels */
    int font_size;            /**< Font size in points */
} feynman_options_t;

/* ============================================================================
 * DIAGRAM CREATION AND MANAGEMENT
 * ============================================================================ */

/**
 * Create a new Feynman diagram
 *
 * @param process Process notation (e.g., "e+ e- -> mu+ mu-")
 * @return Pointer to new diagram, or NULL on failure
 */
feynman_diagram_t *feynman_create(const char *process);

/**
 * Free Feynman diagram and all resources
 *
 * @param diagram Diagram to free
 */
void feynman_free(feynman_diagram_t *diagram);

/**
 * Set diagram title
 *
 * @param diagram Diagram to modify
 * @param title Title string
 */
void feynman_set_title(feynman_diagram_t *diagram, const char *title);

/**
 * Set loop order (0 = tree level, 1 = one-loop, etc.)
 *
 * @param diagram Diagram to modify
 * @param order Loop order
 */
void feynman_set_loop_order(feynman_diagram_t *diagram, int order);

/* ============================================================================
 * ADDING VERTICES
 * ============================================================================ */

/**
 * Add a vertex at specified position
 *
 * @param diagram Diagram to modify
 * @param x X coordinate
 * @param y Y coordinate
 * @return Vertex ID, or -1 on error
 */
int feynman_add_vertex(feynman_diagram_t *diagram, double x, double y);

/**
 * Add a labeled vertex
 *
 * @param diagram Diagram to modify
 * @param x X coordinate
 * @param y Y coordinate
 * @param label Vertex label (e.g., coupling constant)
 * @return Vertex ID, or -1 on error
 */
int feynman_add_vertex_labeled(feynman_diagram_t *diagram, double x, double y,
                               const char *label);

/**
 * Add an external vertex (for incoming/outgoing particles)
 *
 * @param diagram Diagram to modify
 * @param x X coordinate
 * @param y Y coordinate
 * @param label Particle label
 * @return Vertex ID, or -1 on error
 */
int feynman_add_external_vertex(feynman_diagram_t *diagram, double x, double y,
                                const char *label);

/* ============================================================================
 * ADDING PROPAGATORS
 * ============================================================================ */

/**
 * Add a fermion propagator (solid line with forward arrow)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (e.g., "e-", "mu-", "u")
 * @return 0 on success, -1 on error
 */
int feynman_add_fermion(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add an antifermion propagator (solid line with backward arrow)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (e.g., "e+", "mu+")
 * @return 0 on success, -1 on error
 */
int feynman_add_antifermion(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a photon propagator (wavy line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (default "gamma")
 * @return 0 on success, -1 on error
 */
int feynman_add_photon(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a gluon propagator (curly line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (default "g")
 * @return 0 on success, -1 on error
 */
int feynman_add_gluon(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a W boson propagator (dashed line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (e.g., "W+", "W-")
 * @return 0 on success, -1 on error
 */
int feynman_add_w_boson(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a Z boson propagator (dashed line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (default "Z")
 * @return 0 on success, -1 on error
 */
int feynman_add_z_boson(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a Higgs propagator (dotted line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label (default "H")
 * @return 0 on success, -1 on error
 */
int feynman_add_higgs(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a scalar propagator (dashed line)
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param label Particle label
 * @return 0 on success, -1 on error
 */
int feynman_add_scalar(feynman_diagram_t *diagram, int from, int to, const char *label);

/**
 * Add a generic propagator with specified type
 *
 * @param diagram Diagram to modify
 * @param from Starting vertex ID
 * @param to Ending vertex ID
 * @param type Particle type
 * @param label Particle label
 * @return 0 on success, -1 on error
 */
int feynman_add_propagator(feynman_diagram_t *diagram, int from, int to,
                           particle_type_t type, const char *label);

/**
 * Add an external (incoming) line to a vertex
 *
 * @param diagram Diagram to modify
 * @param vertex Target vertex ID
 * @param type Particle type
 * @param label Particle label
 * @param direction Direction angle in degrees (0=right, 90=up, 180=left, 270=down)
 * @return 0 on success, -1 on error
 */
int feynman_add_incoming(feynman_diagram_t *diagram, int vertex, particle_type_t type,
                         const char *label, double direction);

/**
 * Add an external (outgoing) line from a vertex
 *
 * @param diagram Diagram to modify
 * @param vertex Source vertex ID
 * @param type Particle type
 * @param label Particle label
 * @param direction Direction angle in degrees
 * @return 0 on success, -1 on error
 */
int feynman_add_outgoing(feynman_diagram_t *diagram, int vertex, particle_type_t type,
                         const char *label, double direction);

/* ============================================================================
 * STANDARD DIAGRAMS
 * ============================================================================ */

/**
 * Create QED vertex diagram (e- -> e- + gamma)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_qed_vertex(void);

/**
 * Create e+ e- -> mu+ mu- diagram (s-channel)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_ee_to_mumu(void);

/**
 * Create Compton scattering diagram (e- + gamma -> e- + gamma)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_compton(void);

/**
 * Create pair annihilation diagram (e+ e- -> gamma gamma)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_pair_annihilation(void);

/**
 * Create electron self-energy diagram (one-loop)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_electron_self_energy(void);

/**
 * Create vacuum polarization diagram (one-loop photon)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_vacuum_polarization(void);

/**
 * Create Moller scattering diagram (e- e- -> e- e-)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_moller_scattering(void);

/**
 * Create Bhabha scattering diagram (e+ e- -> e+ e-)
 *
 * @return New diagram, or NULL on error
 */
feynman_diagram_t *feynman_create_bhabha_scattering(void);

/* ============================================================================
 * RENDERING OPTIONS
 * ============================================================================ */

/**
 * Get default rendering options
 *
 * @return Default options structure
 */
feynman_options_t feynman_default_options(void);

/**
 * Get publication-quality rendering options
 *
 * @return Publication options structure
 */
feynman_options_t feynman_publication_options(void);

/* ============================================================================
 * ASCII RENDERING
 * ============================================================================ */

/**
 * Render diagram to ASCII string
 *
 * @param diagram Diagram to render
 * @param opts Rendering options (NULL for defaults)
 * @return Allocated string with ASCII diagram (caller must free)
 */
char *feynman_render_ascii(const feynman_diagram_t *diagram, const feynman_options_t *opts);

/**
 * Print diagram to stdout
 *
 * @param diagram Diagram to print
 */
void feynman_print_ascii(const feynman_diagram_t *diagram);

/* ============================================================================
 * SVG RENDERING
 * ============================================================================ */

/**
 * Render diagram to SVG string
 *
 * @param diagram Diagram to render
 * @param width SVG width in pixels
 * @param height SVG height in pixels
 * @return Allocated string with SVG content (caller must free)
 */
char *feynman_render_svg(const feynman_diagram_t *diagram, int width, int height);

/**
 * Save diagram to SVG file
 *
 * @param diagram Diagram to save
 * @param filename Output filename
 * @param width SVG width in pixels
 * @param height SVG height in pixels
 * @return 0 on success, -1 on error
 */
int feynman_save_svg(const feynman_diagram_t *diagram, const char *filename,
                     int width, int height);

/* ============================================================================
 * LATEX RENDERING
 * ============================================================================ */

/**
 * Render diagram to LaTeX/TikZ-Feynman format
 *
 * @param diagram Diagram to render
 * @return Allocated string with LaTeX code (caller must free)
 */
char *feynman_render_latex(const feynman_diagram_t *diagram);

/**
 * Save diagram to LaTeX file
 *
 * @param diagram Diagram to save
 * @param filename Output filename
 * @return 0 on success, -1 on error
 */
int feynman_save_latex(const feynman_diagram_t *diagram, const char *filename);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Get vertex by ID
 *
 * @param diagram Diagram to query
 * @param id Vertex ID
 * @return Pointer to vertex, or NULL if not found
 */
const feynman_vertex_t *feynman_get_vertex(const feynman_diagram_t *diagram, int id);

/**
 * Update diagram bounding box
 *
 * @param diagram Diagram to update
 */
void feynman_update_bounds(feynman_diagram_t *diagram);

/**
 * Get particle type name as string
 *
 * @param type Particle type
 * @return String name (e.g., "fermion", "photon")
 */
const char *feynman_particle_type_name(particle_type_t type);

/**
 * Get TikZ-Feynman style name for particle type
 *
 * @param type Particle type
 * @return TikZ style name (e.g., "fermion", "photon")
 */
const char *feynman_tikz_style(particle_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* FEYNMAN_DIAGRAM_H */
