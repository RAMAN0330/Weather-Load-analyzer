import { AnimatePresence, motion } from 'framer-motion'
import React from 'react'

interface Props {
  children: React.ReactNode
  pageKey: string
}

export const PageTransition = ({ children, pageKey }: Props) => (
  <AnimatePresence mode="wait">
    <motion.div
      key={pageKey}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
      style={{ height: '100%', width: '100%' }}
    >
      {children}
    </motion.div>
  </AnimatePresence>
)
