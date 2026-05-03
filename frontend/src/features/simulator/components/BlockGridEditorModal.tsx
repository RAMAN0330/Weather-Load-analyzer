import React, { useEffect } from 'react';
import { BlockGridEditor } from './BlockGridEditor';

type Props = {
  open: boolean;
  onClose: () => void;
  selectionLabel?: string;
};

export const BlockGridEditorModal: React.FC<Props> = ({ open, onClose, selectionLabel }) => {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="sim-modal-backdrop" onClick={onClose}>
      <div className="sim-modal sim-grid-modal" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <div>
            <h3>Block Grid Editor</h3>
            <span>
              Directly tune block drivers while keeping residuals, exogenous weather, and confidence
              signals in view.
            </span>
          </div>
          <div className="sim-panel-actions">
            {selectionLabel ? <span className="sim-hint-chip">{selectionLabel}</span> : null}
            <button type="button" className="secondary-btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
        <BlockGridEditor />
      </div>
    </div>
  );
};
