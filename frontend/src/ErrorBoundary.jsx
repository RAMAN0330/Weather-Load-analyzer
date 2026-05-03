import React from 'react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unexpected runtime error.' };
  }

  componentDidCatch(error, errorInfo) {
    // Keep this for browser debugging and support diagnostics.
    // eslint-disable-next-line no-console
    console.error('UI runtime error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main
          style={{
            minHeight: '100vh',
            display: 'grid',
            placeItems: 'center',
            background: '#232428',
            color: '#f0e7da',
            padding: '24px',
          }}
        >
          <section
            style={{
              width: 'min(680px, 96vw)',
              border: '1px solid rgba(240, 221, 199, 0.22)',
              borderRadius: 12,
              background: 'rgba(43, 45, 49, 0.9)',
              padding: '16px 18px',
            }}
          >
            <h2 style={{ margin: '0 0 8px 0', fontSize: 18 }}>UI Error</h2>
            <p style={{ margin: '0 0 10px 0', color: '#e1bd9f' }}>
              App hit a runtime error and stopped rendering.
            </p>
            <pre
              style={{
                margin: 0,
                whiteSpace: 'pre-wrap',
                color: '#f0e7da',
                fontSize: 12,
                lineHeight: 1.5,
              }}
            >
              {this.state.message}
            </pre>
            <p style={{ margin: '12px 0 0 0', fontSize: 12, color: '#b9ab96' }}>
              Refresh once. If it repeats, share this message.
            </p>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
