import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { GALLERY_ITEMS } from './galleryData';
import './Gallery.css';

const Gallery: React.FC = () => {
  const logoUrl = `${import.meta.env.BASE_URL}ml-logo.png`;
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const canvasRefs = useRef<Map<string, HTMLCanvasElement>>(new Map());

  const filteredItems = selectedCategory === 'all'
    ? GALLERY_ITEMS
    : GALLERY_ITEMS.filter(item => item.category === selectedCategory);

  const setCanvasRef = useCallback((id: string, el: HTMLCanvasElement | null) => {
    if (el) {
      canvasRefs.current.set(id, el);
    }
  }, []);

  useEffect(() => {
    // Small delay to ensure canvas elements are mounted after filter change
    const timeout = setTimeout(() => {
      filteredItems.forEach(item => {
        const canvas = canvasRefs.current.get(item.id);
        if (canvas) {
          item.render(canvas);
        }
      });
    }, 50);
    return () => clearTimeout(timeout);
  }, [selectedCategory]);

  return (
    <div className="gallery">
      <div className="section-header">
        <img className="section-logo" src={logoUrl} alt="" aria-hidden="true" />
        <div className="section-header-text">
          <h1 className="section-title">Algorithm Gallery</h1>
          <p className="section-description">
            Visual demonstrations of quantum computing concepts and algorithms.
          </p>
        </div>
      </div>

      <div className="gallery-filters">
        <button
          className={`filter-btn ${selectedCategory === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('all')}
        >
          All
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'fundamentals' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('fundamentals')}
        >
          Fundamentals
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'algorithms' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('algorithms')}
        >
          Algorithms
        </button>
        <button
          className={`filter-btn ${selectedCategory === 'applications' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('applications')}
        >
          Applications
        </button>
      </div>

      <div className="gallery-grid">
        {filteredItems.map(item => (
          <Link key={item.id} to={`/gallery/${item.id}`} className="gallery-card-link">
            <div className="gallery-card">
              <canvas
                ref={(el) => setCanvasRef(item.id, el)}
                width={400}
                height={300}
                className="gallery-canvas"
              />
              <div className="gallery-info">
                <h3>{item.title}</h3>
                <p>{item.description}</p>
                <span className="category-tag">{item.category}</span>
              </div>
              <div className="card-overlay">
                <span className="view-text">View Visualization</span>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Gallery;
