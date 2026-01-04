import React, { useRef, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getGalleryItem, GALLERY_ITEMS } from './galleryData';
import './Gallery.css';

const GalleryDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const item = id ? getGalleryItem(id) : undefined;

  useEffect(() => {
    if (!item || !canvasRef.current) return;
    item.render(canvasRef.current);
  }, [item]);

  if (!item) {
    return (
      <div className="gallery-detail">
        <div className="section-header">
          <h1 className="section-title">Visualization Not Found</h1>
          <p className="section-description">
            The requested visualization could not be found.
          </p>
        </div>
        <Link to="/gallery" className="back-link">
          Back to Gallery
        </Link>
      </div>
    );
  }

  // Find related items in the same category
  const relatedItems = GALLERY_ITEMS.filter(
    i => i.category === item.category && i.id !== item.id
  ).slice(0, 3);

  return (
    <div className="gallery-detail">
      <div className="detail-header">
        <Link to="/gallery" className="back-link">
          Back to Gallery
        </Link>
        <span className="category-tag">{item.category}</span>
      </div>

      <div className="section-header">
        <h1 className="section-title">{item.title}</h1>
        <p className="section-description">{item.description}</p>
      </div>

      <div className="visualization-container">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          className="detail-canvas"
        />
      </div>

      {relatedItems.length > 0 && (
        <div className="related-section">
          <h2>Related Visualizations</h2>
          <div className="related-grid">
            {relatedItems.map(related => (
              <Link
                key={related.id}
                to={`/gallery/${related.id}`}
                className="related-card"
              >
                <h3>{related.title}</h3>
                <p>{related.description}</p>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GalleryDetail;
