import React from 'react';

const GlassCard = ({ children, className = '', title = null, subtitle = null, action = null }) => {
  return (
    <div
      className={`bg-[var(--bg-panel)] border border-[color:var(--outline)] rounded-[20px] p-6 ${className} flex flex-col relative overflow-hidden shadow-xl`}
    >
      {/* Header Section */}
      {(title || action) && (
        <div className="flex justify-between items-center mb-6">
          <div>
            {title && (
              <h3 className="text-[color:var(--text)] font-semibold text-lg tracking-tight">{title}</h3>
            )}
            {subtitle && <p className="text-[color:var(--text-muted)] text-xs mt-1">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 relative z-10">{children}</div>
    </div>
  );
};

export default GlassCard;
