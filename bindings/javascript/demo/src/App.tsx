import React, { Suspense, lazy } from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';

const Playground = lazy(() => import('./playground/Playground'));
const Examples = lazy(() => import('./examples/Examples'));
const Gallery = lazy(() => import('./gallery/Gallery'));
const GalleryDetail = lazy(() => import('./gallery/GalleryDetail'));
const Orbitals = lazy(() => import('./orbitals/OrbitalDemo'));

export const App: React.FC = () => {
  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">Q</span>
            <span className="logo-text">Moonlab</span>
          </div>
          <nav className="nav">
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Playground
            </NavLink>
            <NavLink to="/examples" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Examples
            </NavLink>
            <NavLink to="/gallery" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Gallery
            </NavLink>
            <NavLink to="/orbitals" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Schrödinger
            </NavLink>
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
            <Route path="/" element={<Playground />} />
            <Route path="/examples" element={<Examples />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/gallery/:id" element={<GalleryDetail />} />
            <Route path="/orbitals" element={<Orbitals />} />
          </Routes>
        </Suspense>
      </main>

      <footer className="footer">
        <p>Moonlab Quantum Simulator - High-performance quantum computing in your browser</p>
        <p className="footer-sub">Powered by WebAssembly &bull; MIT License &bull; Tsotchke Corporation</p>
      </footer>
    </div>
  );
};

const LoadingSpinner: React.FC = () => (
  <div className="loading-container">
    <div className="quantum-spinner">
      <div className="spinner-ring"></div>
      <div className="spinner-core"></div>
    </div>
    <p>Initializing quantum state...</p>
  </div>
);
