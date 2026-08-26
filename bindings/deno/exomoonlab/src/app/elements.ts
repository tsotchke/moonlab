/**
 * The element picker's list.
 *
 * The web build offers the whole periodic table; this is the first four
 * periods, which is where the shell structure the corrections model is
 * actually visible. Z is what reaches the wavefunction -- everything else is
 * labelling.
 */

export interface Element {
  readonly z: number;
  readonly symbol: string;
  readonly name: string;
}

export const ELEMENTS: readonly Element[] = [
  { z: 1, symbol: "H", name: "Hydrogen" },
  { z: 2, symbol: "He", name: "Helium" },
  { z: 3, symbol: "Li", name: "Lithium" },
  { z: 4, symbol: "Be", name: "Beryllium" },
  { z: 5, symbol: "B", name: "Boron" },
  { z: 6, symbol: "C", name: "Carbon" },
  { z: 7, symbol: "N", name: "Nitrogen" },
  { z: 8, symbol: "O", name: "Oxygen" },
  { z: 9, symbol: "F", name: "Fluorine" },
  { z: 10, symbol: "Ne", name: "Neon" },
  { z: 11, symbol: "Na", name: "Sodium" },
  { z: 12, symbol: "Mg", name: "Magnesium" },
  { z: 13, symbol: "Al", name: "Aluminium" },
  { z: 14, symbol: "Si", name: "Silicon" },
  { z: 15, symbol: "P", name: "Phosphorus" },
  { z: 16, symbol: "S", name: "Sulfur" },
  { z: 17, symbol: "Cl", name: "Chlorine" },
  { z: 18, symbol: "Ar", name: "Argon" },
  { z: 19, symbol: "K", name: "Potassium" },
  { z: 20, symbol: "Ca", name: "Calcium" },
  { z: 26, symbol: "Fe", name: "Iron" },
  { z: 29, symbol: "Cu", name: "Copper" },
  { z: 30, symbol: "Zn", name: "Zinc" },
  { z: 36, symbol: "Kr", name: "Krypton" },
];
