import React from 'react';
import type { Atom } from './elements';

interface ElementPickerProps {
  elements: Atom[];
  isOpen: boolean;
  onClose: () => void;
  onSelect: (atom: Atom) => void;
  selected?: Atom;
}

const CATEGORY_CLASS: Record<Atom['category'], string> = {
  alkali: 'cat-alkali',
  alkaline: 'cat-alkaline',
  transition: 'cat-transition',
  'post-transition': 'cat-post',
  metalloid: 'cat-metalloid',
  nonmetal: 'cat-nonmetal',
  halogen: 'cat-halogen',
  noble: 'cat-noble',
  lanthanide: 'cat-lanthanide',
  actinide: 'cat-actinide',
};

export const ElementPicker: React.FC<ElementPickerProps> = ({
  elements,
  isOpen,
  onClose,
  onSelect,
  selected,
}) => {
  if (!isOpen) return null;

  const sorted = [...elements].sort((a, b) => a.Z - b.Z);

  return (
    <div className="element-picker-overlay" role="dialog" aria-modal="true">
      <div className="element-picker">
        <div className="picker-header">
          <div>
            <div className="pill">Periodic Table</div>
            <h3>Select an element</h3>
            <p>Hover to see the name, click to choose.</p>
          </div>
          <button className="btn btn-secondary" onClick={onClose} aria-label="Close element picker">
            × Close
          </button>
        </div>

        <div className="element-grid">
          {sorted.map((el) => (
            <button
              key={el.Z}
              className={`element-cell ${CATEGORY_CLASS[el.category]} ${
                selected?.symbol === el.symbol ? 'selected' : ''
              }`}
              style={{ gridColumn: el.group, gridRow: el.period }}
              title={`${el.name} (Z=${el.Z})`}
              onClick={() => {
                onSelect(el);
                onClose();
              }}
            >
              <span className="element-number">{el.Z}</span>
              <span className="element-symbol">{el.symbol}</span>
            </button>
          ))}
        </div>

        <div className="picker-legend">
          <span className="legend-chip cat-alkali">Alkali</span>
          <span className="legend-chip cat-alkaline">Alkaline</span>
          <span className="legend-chip cat-transition">Transition</span>
          <span className="legend-chip cat-post">Post-transition</span>
          <span className="legend-chip cat-metalloid">Metalloid</span>
          <span className="legend-chip cat-nonmetal">Nonmetal</span>
          <span className="legend-chip cat-halogen">Halogen</span>
          <span className="legend-chip cat-noble">Noble gas</span>
          <span className="legend-chip cat-lanthanide">Lanthanide</span>
          <span className="legend-chip cat-actinide">Actinide</span>
        </div>
      </div>
    </div>
  );
};

export default ElementPicker;
