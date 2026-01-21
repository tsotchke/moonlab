import React, { Suspense, lazy, useEffect, useState } from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';

const Playground = lazy(() => import('./playground/Playground'));
const Examples = lazy(() => import('./examples/Examples'));
const Gallery = lazy(() => import('./gallery/Gallery'));
const GalleryDetail = lazy(() => import('./gallery/GalleryDetail'));
const Orbitals = lazy(() => import('./orbitals/OrbitalDemo'));

export const App: React.FC = () => {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const logoUrl = `${import.meta.env.BASE_URL}moonlab.png`;
  const moonBgUrl = `${import.meta.env.BASE_URL}moon-dark-lowrez.png`;

  useEffect(() => {
    document.documentElement.style.setProperty('--moon-bg-image', `url("${moonBgUrl}")`);
  }, [moonBgUrl]);

  const closeNav = () => setIsNavOpen(false);

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <img className="logo-icon" src={logoUrl} alt="Moonlab logo" />
            <span className="logo-text">Moonlab</span>
          </div>
          <button
            className="menu-toggle"
            type="button"
            aria-label={isNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={isNavOpen}
            aria-controls="primary-nav"
            onClick={() => setIsNavOpen((prev) => !prev)}
          >
            <span className="menu-bar"></span>
            <span className="menu-bar"></span>
            <span className="menu-bar"></span>
          </button>
          <nav className={`nav ${isNavOpen ? 'open' : ''}`} id="primary-nav" aria-label="Primary">
            <NavLink
              to="/playground"
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              onClick={closeNav}
            >
              Playground
            </NavLink>
            <NavLink
              to="/examples"
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              onClick={closeNav}
            >
              Examples
            </NavLink>
            <NavLink
              to="/gallery"
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              onClick={closeNav}
            >
              Gallery
            </NavLink>
            <NavLink
              to="/schrodinger"
              className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
              onClick={closeNav}
            >
              Schrödinger
            </NavLink>
            <a
              href="https://github.com/tsotchke/moonlab"
              className="nav-link nav-link-external"
              target="_blank"
              rel="noopener noreferrer"
              onClick={closeNav}
            >
              GitHub
            </a>
          </nav>
          <a
            href="https://github.com/tsotchke/moonlab"
            className="github-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </div>
      </header>

      <main className="main">
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<Orbitals />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/examples" element={<Examples />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/gallery/:id" element={<GalleryDetail />} />
            <Route path="/schrodinger" element={<Orbitals />} />
          </Routes>
        </Suspense>
      </main>

      <footer className="footer">
        <div className="footer-logo-row">
          <img className="footer-logo" src={logoUrl} alt="Moonlab logo" />
          <p>Moonlab Quantum Simulator - High-performance quantum computing in your browser</p>
        </div>
        <p className="footer-sub">Powered by WebAssembly &bull; MIT License &bull; Tsotchke Corporation</p>
      </footer>
    </div>
  );
};

const LoadingSpinner: React.FC = () => (
  <div className="loading-container">
    <img
      className="loading-gif"
      src={`${import.meta.env.BASE_URL}moonlab_glitch.gif`}
      alt="Moonlab loading animation"
    />
    <p>Initializing quantum state...</p>
  </div>
);
