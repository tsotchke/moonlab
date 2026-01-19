import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { App } from './App';
import './styles/index.css';

const basePath =
  typeof document !== 'undefined'
    ? new URL(document.baseURI).pathname
    : import.meta.env.BASE_URL;
const baseDir = basePath.endsWith('/') ? basePath : basePath.replace(/\/[^/]*$/, '/');
const routerBase = baseDir.replace(/\/$/, '') || '/';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter basename={routerBase}>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
