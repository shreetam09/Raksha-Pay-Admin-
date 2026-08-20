import { createRoot } from 'react-dom/client';

import App from './App';
import { ErrorBoundary } from './components/error-boundary';
import appConfig from './data/appConfig.json';

import './index.css';

// Set document title dynamically from appConfig
if (typeof document !== 'undefined' && appConfig.branding?.name) {
  document.title = `${appConfig.branding.name} Admin`;
}

createRoot(document.getElementById('root'), {
  // Keeps caught errors off reportError(), which would raise the dev overlay.
  onCaughtError: (error, errorInfo) => {
    console.error(error, errorInfo.componentStack);
  },
}).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
);
