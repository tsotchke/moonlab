/**
 * @file lattice_2d.c
 * @brief Implementation of 2D lattice support for MPS simulations
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "lattice_2d.h"
#include <stdio.h>
#include <string.h>

// ============================================================================
// LATTICE CREATION
// ============================================================================

/**
 * @brief Compute neighbors for square lattice
 */
static void compute_square_neighbors(lattice_2d_t *lat) {
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);
        uint32_t n = 0;

        // Right neighbor (+x)
        int32_t nx = c.x + 1;
        int32_t ny = c.y;
        if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
            nx = nx % (int32_t)lat->Lx;
        }
        if (nx >= 0 && (uint32_t)nx < lat->Lx) {
            lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
            lat->neighbors[s][n].bond = (bond_vector_t){1.0, 0.0, 0.0};
            lat->neighbors[s][n].distance = 1.0;
            lat->neighbors[s][n].valid = true;
            n++;
        }

        // Left neighbor (-x)
        nx = c.x - 1;
        ny = c.y;
        if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
            nx = (nx + lat->Lx) % lat->Lx;
        }
        if (nx >= 0 && (uint32_t)nx < lat->Lx) {
            lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
            lat->neighbors[s][n].bond = (bond_vector_t){-1.0, 0.0, 0.0};
            lat->neighbors[s][n].distance = 1.0;
            lat->neighbors[s][n].valid = true;
            n++;
        }

        // Up neighbor (+y)
        nx = c.x;
        ny = c.y + 1;
        if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
            ny = ny % (int32_t)lat->Ly;
        }
        if (ny >= 0 && (uint32_t)ny < lat->Ly) {
            lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
            lat->neighbors[s][n].bond = (bond_vector_t){0.0, 1.0, 0.0};
            lat->neighbors[s][n].distance = 1.0;
            lat->neighbors[s][n].valid = true;
            n++;
        }

        // Down neighbor (-y)
        nx = c.x;
        ny = c.y - 1;
        if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
            ny = (ny + lat->Ly) % lat->Ly;
        }
        if (ny >= 0 && (uint32_t)ny < lat->Ly) {
            lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
            lat->neighbors[s][n].bond = (bond_vector_t){0.0, -1.0, 0.0};
            lat->neighbors[s][n].distance = 1.0;
            lat->neighbors[s][n].valid = true;
            n++;
        }

        lat->num_neighbors[s] = n;

        // Mark remaining as invalid
        for (; n < lat->max_neighbors; n++) {
            lat->neighbors[s][n].valid = false;
        }
    }
}

/**
 * @brief Compute neighbors for triangular lattice
 */
static void compute_triangular_neighbors(lattice_2d_t *lat) {
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);
        uint32_t n = 0;

        // 6 neighbors for triangular lattice
        // Directions: (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)
        int dx[] = {1, -1, 0, 0, 1, -1};
        int dy[] = {0, 0, 1, -1, 1, -1};
        double bond_x[] = {1.0, -1.0, 0.5, -0.5, 0.5, -0.5};
        double bond_y[] = {0.0, 0.0, sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2, -sqrt(3)/2};

        for (int i = 0; i < 6; i++) {
            int32_t nx = c.x + dx[i];
            int32_t ny = c.y + dy[i];

            // Apply boundary conditions
            if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
                nx = ((nx % (int32_t)lat->Lx) + lat->Lx) % lat->Lx;
            }
            if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
                ny = ((ny % (int32_t)lat->Ly) + lat->Ly) % lat->Ly;
            }

            if (nx >= 0 && (uint32_t)nx < lat->Lx &&
                ny >= 0 && (uint32_t)ny < lat->Ly) {
                lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
                lat->neighbors[s][n].bond = (bond_vector_t){bond_x[i], bond_y[i], 0.0};
                lat->neighbors[s][n].distance = 1.0;
                lat->neighbors[s][n].valid = true;
                n++;
            }
        }

        lat->num_neighbors[s] = n;

        for (; n < lat->max_neighbors; n++) {
            lat->neighbors[s][n].valid = false;
        }
    }
}

/**
 * @brief Compute neighbors for honeycomb lattice
 *
 * Honeycomb is bipartite with A and B sublattices. Each site has 3 neighbors.
 * A sites at even (x+y), B sites at odd (x+y).
 * A sites: neighbors at (+1,0), (0,+1), (0,-1) or similar pattern
 * B sites: neighbors at (-1,0), (0,+1), (0,-1) or similar pattern
 */
static void compute_honeycomb_neighbors(lattice_2d_t *lat) {
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);
        uint32_t n = 0;

        // Determine sublattice: A if (x+y) is even, B if odd
        bool is_sublattice_A = ((c.x + c.y) % 2 == 0);

        // Honeycomb neighbor directions depend on sublattice
        // Using brick-wall representation for simplicity
        int dx[3], dy[3];
        double bond_x[3], bond_y[3];

        if (is_sublattice_A) {
            // A sublattice: neighbors to right, up, down
            dx[0] = 1;  dy[0] = 0;   bond_x[0] = 1.0;  bond_y[0] = 0.0;
            dx[1] = 0;  dy[1] = 1;   bond_x[1] = 0.5;  bond_y[1] = sqrt(3)/2;
            dx[2] = 0;  dy[2] = -1;  bond_x[2] = 0.5;  bond_y[2] = -sqrt(3)/2;
        } else {
            // B sublattice: neighbors to left, up, down
            dx[0] = -1; dy[0] = 0;   bond_x[0] = -1.0; bond_y[0] = 0.0;
            dx[1] = 0;  dy[1] = 1;   bond_x[1] = -0.5; bond_y[1] = sqrt(3)/2;
            dx[2] = 0;  dy[2] = -1;  bond_x[2] = -0.5; bond_y[2] = -sqrt(3)/2;
        }

        for (int i = 0; i < 3; i++) {
            int32_t nx = c.x + dx[i];
            int32_t ny = c.y + dy[i];

            // Apply boundary conditions
            if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
                nx = ((nx % (int32_t)lat->Lx) + lat->Lx) % lat->Lx;
            }
            if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
                ny = ((ny % (int32_t)lat->Ly) + lat->Ly) % lat->Ly;
            }

            if (nx >= 0 && (uint32_t)nx < lat->Lx &&
                ny >= 0 && (uint32_t)ny < lat->Ly) {
                lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
                lat->neighbors[s][n].bond = (bond_vector_t){bond_x[i], bond_y[i], 0.0};
                lat->neighbors[s][n].distance = 1.0;
                lat->neighbors[s][n].valid = true;
                n++;
            }
        }

        lat->num_neighbors[s] = n;

        for (; n < lat->max_neighbors; n++) {
            lat->neighbors[s][n].valid = false;
        }
    }
}

/**
 * @brief Compute neighbors for Kagome lattice
 *
 * Kagome lattice has 3 sites per unit cell, each with 4 neighbors.
 * It consists of corner-sharing triangles.
 * We map it to a rectangular grid where site_type = s % 3.
 */
static void compute_kagome_neighbors(lattice_2d_t *lat) {
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);
        uint32_t n = 0;

        // Kagome has 3 basis sites per unit cell
        // Site type: 0, 1, or 2 based on position in the unit cell
        // Using column index to determine site type
        int site_type = c.x % 3;

        // Kagome neighbor patterns (4 neighbors per site)
        int dx[4], dy[4];
        double bond_x[4], bond_y[4];

        switch (site_type) {
            case 0:  // Corner site
                dx[0] = 1;  dy[0] = 0;   bond_x[0] = 0.5;  bond_y[0] = 0.0;
                dx[1] = -1; dy[1] = 0;   bond_x[1] = -0.5; bond_y[1] = 0.0;
                dx[2] = 0;  dy[2] = 1;   bond_x[2] = 0.25; bond_y[2] = sqrt(3)/4;
                dx[3] = 0;  dy[3] = -1;  bond_x[3] = 0.25; bond_y[3] = -sqrt(3)/4;
                break;
            case 1:  // Right-triangle site
                dx[0] = -1; dy[0] = 0;   bond_x[0] = -0.5; bond_y[0] = 0.0;
                dx[1] = 1;  dy[1] = 0;   bond_x[1] = 0.5;  bond_y[1] = 0.0;
                dx[2] = 0;  dy[2] = 1;   bond_x[2] = -0.25; bond_y[2] = sqrt(3)/4;
                dx[3] = -1; dy[3] = 1;   bond_x[3] = -0.75; bond_y[3] = sqrt(3)/4;
                break;
            case 2:  // Left-triangle site
                dx[0] = 1;  dy[0] = 0;   bond_x[0] = 0.5;  bond_y[0] = 0.0;
                dx[1] = -1; dy[1] = 0;   bond_x[1] = -0.5; bond_y[1] = 0.0;
                dx[2] = 0;  dy[2] = -1;  bond_x[2] = 0.25; bond_y[2] = -sqrt(3)/4;
                dx[3] = 1;  dy[3] = -1;  bond_x[3] = 0.75; bond_y[3] = -sqrt(3)/4;
                break;
        }

        for (int i = 0; i < 4; i++) {
            int32_t nx = c.x + dx[i];
            int32_t ny = c.y + dy[i];

            // Apply boundary conditions
            if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
                nx = ((nx % (int32_t)lat->Lx) + lat->Lx) % lat->Lx;
            }
            if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
                ny = ((ny % (int32_t)lat->Ly) + lat->Ly) % lat->Ly;
            }

            if (nx >= 0 && (uint32_t)nx < lat->Lx &&
                ny >= 0 && (uint32_t)ny < lat->Ly) {
                lat->neighbors[s][n].site = coord_to_snake(lat, nx, ny);
                lat->neighbors[s][n].bond = (bond_vector_t){bond_x[i], bond_y[i], 0.0};
                lat->neighbors[s][n].distance = 1.0;
                lat->neighbors[s][n].valid = true;
                n++;
            }
        }

        lat->num_neighbors[s] = n;

        for (; n < lat->max_neighbors; n++) {
            lat->neighbors[s][n].valid = false;
        }
    }
}

lattice_2d_t *lattice_2d_create(uint32_t Lx, uint32_t Ly,
                                 lattice_type_t type,
                                 boundary_condition_t bc) {
    if (Lx < 2 || Ly < 2) return NULL;

    lattice_2d_t *lat = (lattice_2d_t *)calloc(1, sizeof(lattice_2d_t));
    if (!lat) return NULL;

    lat->Lx = Lx;
    lat->Ly = Ly;
    lat->num_sites = Lx * Ly;
    lat->type = type;
    lat->bc = bc;
    lat->a = 1.0;  // Default lattice constant

    lat->max_neighbors = lattice_coordination(type);
    if (type == LATTICE_TRIANGULAR) lat->max_neighbors = 6;

    // Allocate mappings
    lat->snake_to_grid = (uint32_t *)malloc(lat->num_sites * sizeof(uint32_t));
    lat->grid_to_snake = (uint32_t *)malloc(lat->num_sites * sizeof(uint32_t));
    lat->snake_to_coord = (coord_2d_t *)malloc(lat->num_sites * sizeof(coord_2d_t));
    lat->num_neighbors = (uint32_t *)calloc(lat->num_sites, sizeof(uint32_t));

    if (!lat->snake_to_grid || !lat->grid_to_snake ||
        !lat->snake_to_coord || !lat->num_neighbors) {
        lattice_2d_free(lat);
        return NULL;
    }

    // Precompute coordinate mappings
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);
        lat->snake_to_coord[s] = c;
        lat->snake_to_grid[s] = coord_to_grid(lat, c.x, c.y);
        lat->grid_to_snake[c.y * Lx + c.x] = s;
    }

    // Allocate neighbor lists
    lat->neighbors = (neighbor_t **)malloc(lat->num_sites * sizeof(neighbor_t *));
    if (!lat->neighbors) {
        lattice_2d_free(lat);
        return NULL;
    }

    for (uint32_t s = 0; s < lat->num_sites; s++) {
        lat->neighbors[s] = (neighbor_t *)calloc(lat->max_neighbors, sizeof(neighbor_t));
        if (!lat->neighbors[s]) {
            lattice_2d_free(lat);
            return NULL;
        }
    }

    // Compute neighbors based on lattice type
    switch (type) {
        case LATTICE_SQUARE:
            compute_square_neighbors(lat);
            break;
        case LATTICE_TRIANGULAR:
            compute_triangular_neighbors(lat);
            break;
        case LATTICE_HONEYCOMB:
            compute_honeycomb_neighbors(lat);
            break;
        case LATTICE_KAGOME:
            compute_kagome_neighbors(lat);
            break;
    }

    return lat;
}

void lattice_2d_free(lattice_2d_t *lat) {
    if (!lat) return;

    if (lat->neighbors) {
        for (uint32_t s = 0; s < lat->num_sites; s++) {
            free(lat->neighbors[s]);
        }
        free(lat->neighbors);
    }

    free(lat->snake_to_grid);
    free(lat->grid_to_snake);
    free(lat->snake_to_coord);
    free(lat->num_neighbors);
    free(lat);
}

// ============================================================================
// NEIGHBOR FUNCTIONS
// ============================================================================

uint32_t lattice_2d_get_neighbors(const lattice_2d_t *lat,
                                   uint32_t snake_idx,
                                   neighbor_t *neighbors) {
    if (!lat || snake_idx >= lat->num_sites || !neighbors) return 0;

    uint32_t count = lat->num_neighbors[snake_idx];
    memcpy(neighbors, lat->neighbors[snake_idx], count * sizeof(neighbor_t));
    return count;
}

uint32_t lattice_2d_neighbor_at(const lattice_2d_t *lat,
                                 uint32_t snake_idx,
                                 int dx, int dy) {
    if (!lat || snake_idx >= lat->num_sites) return UINT32_MAX;

    coord_2d_t c = lat->snake_to_coord[snake_idx];
    int32_t nx = c.x + dx;
    int32_t ny = c.y + dy;

    // Apply boundary conditions
    if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
        nx = ((nx % (int32_t)lat->Lx) + lat->Lx) % lat->Lx;
    }
    if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
        ny = ((ny % (int32_t)lat->Ly) + lat->Ly) % lat->Ly;
    }

    if (nx < 0 || (uint32_t)nx >= lat->Lx ||
        ny < 0 || (uint32_t)ny >= lat->Ly) {
        return UINT32_MAX;
    }

    return coord_to_snake(lat, nx, ny);
}

bool lattice_2d_are_neighbors(const lattice_2d_t *lat,
                               uint32_t site1, uint32_t site2) {
    if (!lat || site1 >= lat->num_sites || site2 >= lat->num_sites) {
        return false;
    }

    for (uint32_t n = 0; n < lat->num_neighbors[site1]; n++) {
        if (lat->neighbors[site1][n].site == site2) {
            return true;
        }
    }
    return false;
}

bond_vector_t lattice_2d_bond_vector(const lattice_2d_t *lat,
                                      uint32_t from, uint32_t to) {
    bond_vector_t bv = {0, 0, 0};
    if (!lat || from >= lat->num_sites || to >= lat->num_sites) {
        return bv;
    }

    for (uint32_t n = 0; n < lat->num_neighbors[from]; n++) {
        if (lat->neighbors[from][n].site == to) {
            return lat->neighbors[from][n].bond;
        }
    }

    // Not direct neighbors - compute from coordinates
    coord_2d_t c1 = lat->snake_to_coord[from];
    coord_2d_t c2 = lat->snake_to_coord[to];

    double dx = c2.x - c1.x;
    double dy = c2.y - c1.y;
    double dist = sqrt(dx*dx + dy*dy);

    if (dist > 0) {
        bv.dx = dx / dist;
        bv.dy = dy / dist;
    }

    return bv;
}

// ============================================================================
// DISTANCE AND GEOMETRY
// ============================================================================

double lattice_2d_distance(const lattice_2d_t *lat,
                            uint32_t site1, uint32_t site2) {
    if (!lat || site1 >= lat->num_sites || site2 >= lat->num_sites) {
        return -1.0;
    }

    coord_2d_t c1 = lat->snake_to_coord[site1];
    coord_2d_t c2 = lat->snake_to_coord[site2];

    double dx = c2.x - c1.x;
    double dy = c2.y - c1.y;

    // Minimum image convention for periodic boundaries
    if (lat->bc == BC_PERIODIC_X || lat->bc == BC_PERIODIC_XY) {
        if (dx > (int32_t)lat->Lx / 2) dx -= lat->Lx;
        if (dx < -(int32_t)lat->Lx / 2) dx += lat->Lx;
    }
    if (lat->bc == BC_PERIODIC_Y || lat->bc == BC_PERIODIC_XY) {
        if (dy > (int32_t)lat->Ly / 2) dy -= lat->Ly;
        if (dy < -(int32_t)lat->Ly / 2) dy += lat->Ly;
    }

    return sqrt(dx*dx + dy*dy) * lat->a;
}

void lattice_2d_real_coords(const lattice_2d_t *lat,
                             uint32_t snake_idx,
                             double *x, double *y) {
    if (!lat || snake_idx >= lat->num_sites || !x || !y) return;

    coord_2d_t c = lat->snake_to_coord[snake_idx];

    switch (lat->type) {
        case LATTICE_TRIANGULAR:
            // Triangular: stagger every other row
            *x = c.x * lat->a + (c.y % 2) * 0.5 * lat->a;
            *y = c.y * lat->a * sqrt(3.0) / 2.0;
            break;

        case LATTICE_HONEYCOMB:
            // Honeycomb: two-atom unit cell
            *x = c.x * lat->a * 1.5;
            *y = c.y * lat->a * sqrt(3.0);
            if ((c.x + c.y) % 2 == 1) {
                *y += lat->a * sqrt(3.0) / 2.0;
            }
            break;

        case LATTICE_SQUARE:
        case LATTICE_KAGOME:
        default:
            *x = c.x * lat->a;
            *y = c.y * lat->a;
            break;
    }
}

// ============================================================================
// VISUALIZATION
// ============================================================================

void lattice_2d_print_info(const lattice_2d_t *lat) {
    if (!lat) return;

    const char *type_names[] = {"Square", "Triangular", "Honeycomb", "Kagome"};
    const char *bc_names[] = {"Open", "Periodic-X", "Periodic-Y", "Periodic-XY"};

    printf("2D Lattice Configuration:\n");
    printf("  Dimensions:     %u x %u = %u sites\n", lat->Lx, lat->Ly, lat->num_sites);
    printf("  Type:           %s\n", type_names[lat->type]);
    printf("  Coordination:   %u\n", lat->max_neighbors);
    printf("  Boundaries:     %s\n", bc_names[lat->bc]);
    printf("  Lattice const:  %.3f\n", lat->a);
}

void lattice_2d_print_snake(const lattice_2d_t *lat) {
    if (!lat) return;

    printf("\nSnake ordering (MPS index at each grid position):\n");
    for (int32_t y = lat->Ly - 1; y >= 0; y--) {
        printf("  ");
        for (uint32_t x = 0; x < lat->Lx; x++) {
            uint32_t s = coord_to_snake(lat, x, y);
            printf("%3u ", s);
        }
        // Show arrow direction
        if (y % 2 == 0) {
            printf(" →");
        } else {
            printf(" ←");
        }
        printf("\n");
    }
    printf("\n");
}

// ============================================================================
// SKYRMION UTILITIES
// ============================================================================

double compute_plaquette_charge(const double *Si, const double *Sj, const double *Sk) {
    // Solid angle formula: Ω = 2 * atan2(Si · (Sj × Sk), 1 + Si·Sj + Sj·Sk + Sk·Si)
    // Topological charge contribution: q = Ω / (4π)

    // Cross product Sj × Sk
    double cross_x = Sj[1]*Sk[2] - Sj[2]*Sk[1];
    double cross_y = Sj[2]*Sk[0] - Sj[0]*Sk[2];
    double cross_z = Sj[0]*Sk[1] - Sj[1]*Sk[0];

    // Si · (Sj × Sk)
    double triple = Si[0]*cross_x + Si[1]*cross_y + Si[2]*cross_z;

    // Dot products
    double Si_Sj = Si[0]*Sj[0] + Si[1]*Sj[1] + Si[2]*Sj[2];
    double Sj_Sk = Sj[0]*Sk[0] + Sj[1]*Sk[1] + Sj[2]*Sk[2];
    double Sk_Si = Sk[0]*Si[0] + Sk[1]*Si[1] + Sk[2]*Si[2];

    double denom = 1.0 + Si_Sj + Sj_Sk + Sk_Si;

    // Avoid division by zero
    if (fabs(denom) < 1e-10) {
        return (triple >= 0) ? 0.25 : -0.25;  // Half a skyrmion
    }

    double omega = 2.0 * atan2(triple, denom);
    return omega / (4.0 * M_PI);
}

uint32_t (*lattice_2d_get_plaquettes(const lattice_2d_t *lat,
                                      uint32_t *num_plaquettes))[3] {
    if (!lat || !num_plaquettes) return NULL;

    // For square lattice: each unit cell has 2 triangular plaquettes
    // Total plaquettes ~ 2 * (Lx-1) * (Ly-1) for open boundaries
    uint32_t np = 2 * (lat->Lx - 1) * (lat->Ly - 1);
    *num_plaquettes = np;

    uint32_t (*plaq)[3] = malloc(np * sizeof(uint32_t[3]));
    if (!plaq) return NULL;

    uint32_t p = 0;
    for (uint32_t y = 0; y < lat->Ly - 1; y++) {
        for (uint32_t x = 0; x < lat->Lx - 1; x++) {
            // Lower-left triangle
            plaq[p][0] = coord_to_snake(lat, x, y);
            plaq[p][1] = coord_to_snake(lat, x+1, y);
            plaq[p][2] = coord_to_snake(lat, x, y+1);
            p++;

            // Upper-right triangle
            plaq[p][0] = coord_to_snake(lat, x+1, y);
            plaq[p][1] = coord_to_snake(lat, x+1, y+1);
            plaq[p][2] = coord_to_snake(lat, x, y+1);
            p++;
        }
    }

    return plaq;
}

void skyrmion_init_classical(const lattice_2d_t *lat,
                              double cx, double cy,
                              double radius, double helicity,
                              int polarity, double (*spins)[3]) {
    if (!lat || !spins) return;

    for (uint32_t s = 0; s < lat->num_sites; s++) {
        double x, y;
        lattice_2d_real_coords(lat, s, &x, &y);

        // Distance from skyrmion center
        double dx = x - cx;
        double dy = y - cy;
        double r = sqrt(dx*dx + dy*dy);

        // Angle around skyrmion
        double phi = atan2(dy, dx);

        // Skyrmion profile: theta goes from 0 at center to pi at edge
        // Using simple profile: theta = pi * min(r/R, 1)
        double theta = M_PI * (r < radius ? r / radius : 1.0);

        // Apply polarity
        if (polarity < 0) {
            theta = M_PI - theta;
        }

        // Spin components (spherical to Cartesian)
        // Helicity determines whether Néel (0) or Bloch (π/2)
        spins[s][0] = sin(theta) * cos(phi + helicity);  // Sx
        spins[s][1] = sin(theta) * sin(phi + helicity);  // Sy
        spins[s][2] = cos(theta);                        // Sz
    }
}
