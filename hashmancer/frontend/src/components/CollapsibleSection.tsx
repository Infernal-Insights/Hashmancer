import React, { useState, ReactNode } from 'react'

interface CollapsibleSectionProps {
  title: string
  children: ReactNode
  defaultExpanded?: boolean
  className?: string
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  defaultExpanded = true,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <div className={`hashmancer-section ${!isExpanded ? 'collapsed' : ''} ${className}`}>
      <div className="section-header" onClick={toggleExpanded}>
        <h2 className="section-title">{title}</h2>
        <button className="section-toggle" type="button">
          {isExpanded ? '[-]' : '[+]'}
        </button>
      </div>
      {isExpanded && (
        <div className="section-content">
          {children}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection