export interface GalleryItem {
  id: string;
  title: string;
  description: string;
  category: 'fundamentals' | 'algorithms' | 'applications';
  render: (canvas: HTMLCanvasElement) => void;
}

export const GALLERY_ITEMS: GalleryItem[] = [
  {
    id: 'bell-states',
    title: 'Bell States Visualization',
    description: 'The four maximally entangled Bell states and their probability distributions',
    category: 'fundamentals',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      const bellStates = [
        { name: '|Φ+⟩', probs: [0.5, 0, 0, 0.5], color: '#6496ff' },
        { name: '|Φ-⟩', probs: [0.5, 0, 0, 0.5], color: '#9966ff' },
        { name: '|Ψ+⟩', probs: [0, 0.5, 0.5, 0], color: '#00d4ff' },
        { name: '|Ψ-⟩', probs: [0, 0.5, 0.5, 0], color: '#ff66aa' },
      ];

      const cellW = w / 2;
      const cellH = h / 2;

      bellStates.forEach((state, idx) => {
        const col = idx % 2;
        const row = Math.floor(idx / 2);
        const cx = col * cellW + cellW / 2;
        const cy = row * cellH + cellH / 2;

        // Draw state name
        ctx.fillStyle = state.color;
        ctx.font = 'bold 18px serif';
        ctx.textAlign = 'center';
        ctx.fillText(state.name, cx, cy - 50);

        // Draw probability bars
        const barW = 25;
        const maxH = 60;
        const labels = ['00', '01', '10', '11'];

        state.probs.forEach((p, i) => {
          const bx = cx - 60 + i * (barW + 8);
          const barH = p * maxH;

          ctx.fillStyle = state.color;
          ctx.fillRect(bx, cy + 30 - barH, barW, barH);

          ctx.fillStyle = '#606080';
          ctx.font = '10px monospace';
          ctx.fillText(`|${labels[i]}⟩`, bx + barW / 2, cy + 45);
        });
      });
    },
  },
  {
    id: 'grover-amplification',
    title: "Grover's Amplitude Amplification",
    description: 'Visualization of how Grover iterations amplify the marked state probability',
    category: 'algorithms',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      // Simulate Grover amplification
      const N = 8;
      const marked = 5;
      const iterations = 6;

      const cellW = w / iterations;

      for (let iter = 0; iter < iterations; iter++) {
        const x = iter * cellW + cellW / 2;

        // Calculate amplitudes after iter iterations
        const theta = Math.asin(1 / Math.sqrt(N));
        const angle = (2 * iter + 1) * theta;
        const markedAmp = Math.sin(angle);
        const otherAmp = Math.cos(angle) / Math.sqrt(N - 1);

        // Draw bars
        const barW = 8;
        const maxH = h - 100;

        for (let i = 0; i < N; i++) {
          const amp = i === marked ? markedAmp : otherAmp;
          const prob = amp * amp;
          const barH = prob * maxH;

          const bx = x - (N * barW) / 2 + i * barW;
          const by = h - 50 - barH;

          ctx.fillStyle = i === marked ? '#00ff88' : '#6496ff';
          ctx.fillRect(bx, by, barW - 1, barH);
        }

        // Iteration label
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Iter ${iter}`, x, h - 20);
      }

      // Title
      ctx.fillStyle = '#e0e0e0';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Marked state (green) probability grows with each iteration', w / 2, 30);
    },
  },
  {
    id: 'quantum-interference',
    title: 'Quantum Interference',
    description: 'Constructive and destructive interference patterns in quantum circuits',
    category: 'fundamentals',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      // Draw interference pattern
      const centerY = h / 2;

      for (let x = 0; x < w; x++) {
        const wave1 = Math.sin(x * 0.05);
        const wave2 = Math.sin(x * 0.05 + Math.PI * 0.3);
        const combined = (wave1 + wave2) / 2;

        // Wave 1 (blue)
        ctx.fillStyle = 'rgba(100, 150, 255, 0.5)';
        ctx.fillRect(x, centerY - 80 + wave1 * 30, 1, 2);

        // Wave 2 (purple)
        ctx.fillStyle = 'rgba(153, 102, 255, 0.5)';
        ctx.fillRect(x, centerY + wave2 * 30, 1, 2);

        // Combined (cyan)
        const intensity = Math.abs(combined);
        ctx.fillStyle = `rgba(0, 212, 255, ${intensity})`;
        ctx.fillRect(x, centerY + 80 + combined * 40, 1, 3);
      }

      // Labels
      ctx.fillStyle = '#6496ff';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Wave 1', 20, centerY - 100);

      ctx.fillStyle = '#9966ff';
      ctx.fillText('Wave 2', 20, centerY - 20);

      ctx.fillStyle = '#00d4ff';
      ctx.fillText('Interference', 20, centerY + 60);
    },
  },
  {
    id: 'vqe-landscape',
    title: 'VQE Energy Landscape',
    description: 'Optimization path through the variational energy landscape',
    category: 'applications',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      // Draw energy landscape
      const landscape = (x: number, y: number) => {
        const nx = (x - w / 2) / 100;
        const ny = (y - h / 2) / 100;
        return Math.sin(nx * 2) * Math.cos(ny * 2) + nx * nx * 0.5 + ny * ny * 0.5;
      };

      // Draw contours
      for (let x = 0; x < w; x += 4) {
        for (let y = 0; y < h; y += 4) {
          const e = landscape(x, y);
          const intensity = Math.max(0, 1 - Math.abs(e) * 0.3);

          if (e < 0) {
            ctx.fillStyle = `rgba(0, 255, 136, ${intensity * 0.3})`;
          } else {
            ctx.fillStyle = `rgba(255, 100, 100, ${intensity * 0.3})`;
          }
          ctx.fillRect(x, y, 4, 4);
        }
      }

      // Draw optimization path
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 2;
      ctx.beginPath();

      let px = w * 0.2;
      let py = h * 0.2;
      ctx.moveTo(px, py);

      for (let i = 0; i < 20; i++) {
        // Gradient descent simulation
        const dx = landscape(px + 1, py) - landscape(px - 1, py);
        const dy = landscape(px, py + 1) - landscape(px, py - 1);
        px -= dx * 20;
        py -= dy * 20;
        px = Math.max(20, Math.min(w - 20, px));
        py = Math.max(20, Math.min(h - 20, py));
        ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Draw minimum marker
      ctx.fillStyle = '#00ff88';
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();

      // Legend
      ctx.fillStyle = '#e0e0e0';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Green = Low energy', 20, 25);
      ctx.fillText('Red = High energy', 20, 45);
      ctx.fillStyle = '#ffcc00';
      ctx.fillText('Yellow = Optimization path', 20, 65);
    },
  },
  {
    id: 'qaoa-layers',
    title: 'QAOA Layer Evolution',
    description: 'How QAOA layers progressively improve the solution quality',
    category: 'applications',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      // Simulate QAOA improvement
      const layers = [0, 1, 2, 3, 4, 5];
      const approxRatios = [0.5, 0.68, 0.78, 0.85, 0.90, 0.93];

      const barW = 60;
      const maxH = h - 120;
      const startX = (w - layers.length * (barW + 20)) / 2;

      layers.forEach((p, i) => {
        const x = startX + i * (barW + 20);
        const barH = approxRatios[i] * maxH;

        // Bar gradient
        const gradient = ctx.createLinearGradient(x, h - 60, x, h - 60 - barH);
        gradient.addColorStop(0, '#6496ff');
        gradient.addColorStop(1, '#00d4ff');
        ctx.fillStyle = gradient;
        ctx.fillRect(x, h - 60 - barH, barW, barH);

        // Percentage label
        ctx.fillStyle = '#e0e0e0';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${(approxRatios[i] * 100).toFixed(0)}%`, x + barW / 2, h - 70 - barH);

        // Layer label
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '12px sans-serif';
        ctx.fillText(`p=${p}`, x + barW / 2, h - 35);
      });

      // Title
      ctx.fillStyle = '#e0e0e0';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Approximation Ratio vs QAOA Depth (p)', w / 2, 30);

      // Optimal line
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(startX - 20, h - 60 - maxH);
      ctx.lineTo(startX + layers.length * (barW + 20), h - 60 - maxH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#00ff88';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('Optimal', startX - 25, h - 55 - maxH);
    },
  },
  {
    id: 'entanglement-entropy',
    title: 'Entanglement Entropy',
    description: 'Measuring quantum correlations through von Neumann entropy',
    category: 'fundamentals',
    render: (canvas) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#0d0d1a';
      ctx.fillRect(0, 0, w, h);

      // Draw entropy curve
      const entropy = (p: number) => {
        if (p <= 0 || p >= 1) return 0;
        return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
      };

      ctx.strokeStyle = '#9966ff';
      ctx.lineWidth = 3;
      ctx.beginPath();

      const margin = 60;
      const graphW = w - margin * 2;
      const graphH = h - margin * 2;

      for (let i = 0; i <= 100; i++) {
        const p = i / 100;
        const x = margin + p * graphW;
        const y = margin + (1 - entropy(p)) * graphH;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Mark maximum
      ctx.fillStyle = '#ff66aa';
      ctx.beginPath();
      ctx.arc(margin + 0.5 * graphW, margin, 8, 0, Math.PI * 2);
      ctx.fill();

      // Axes
      ctx.strokeStyle = '#404060';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin, margin);
      ctx.lineTo(margin, h - margin);
      ctx.lineTo(w - margin, h - margin);
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#a0a0b0';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Probability p', w / 2, h - 20);

      ctx.save();
      ctx.translate(20, h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Entropy S(p)', 0, 0);
      ctx.restore();

      ctx.fillStyle = '#e0e0e0';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Max entropy at p=0.5 (Bell state)', w / 2, margin - 25);
    },
  },
];

export function getGalleryItem(id: string): GalleryItem | undefined {
  return GALLERY_ITEMS.find(item => item.id === id);
}
