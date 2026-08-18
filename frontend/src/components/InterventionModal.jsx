import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, PhoneCall, Smartphone, UserPlus } from 'lucide-react';

const riskFactors = [
  {
    id: 'active_call',
    title: 'Active Phone Call',
    description: 'You are on a call while making this high-value transfer',
    icon: PhoneCall,
  },
  {
    id: 'unfamiliar_device',
    title: 'Unfamiliar Device',
    description: 'This transaction is from a device you rarely use',
    icon: Smartphone,
  },
  {
    id: 'first_time_payee',
    title: 'First-Time Payee',
    description: 'You have never sent money to this UPI ID before',
    icon: UserPlus,
  },
];

// Slower, more deliberate stagger animation
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.25, delayChildren: 0.6 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, x: 20 },
  show: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 120, damping: 20 } }
};

export const InterventionModal = ({
  isOpen = true, 
  onCancel,
  onProceed,
  amount = '25,000',
  payee = 'Ramesh Kumar',
  upiId = 'ramesh@upi'
}) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: '20%' }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: '20%' }}
          transition={{ type: 'spring', damping: 30, stiffness: 120 }}
          className="fixed inset-0 z-50 flex flex-col md:flex-row w-full h-[100dvh] bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-obsidian)] overflow-hidden transition-colors duration-300"
        >
          {/* LEFT PANEL: The Urgent Alert */}
          <div className="w-full md:w-5/12 bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] p-6 md:p-12 flex flex-col justify-center items-center relative overflow-hidden shrink-0 transition-colors duration-300">
            {/* Subtle pulse background effect */}
            <motion.div 
              animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.2, 0.1] }} 
              transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
              className="absolute -right-20 -top-20 w-64 h-64 bg-[var(--color-off-black-ink)] dark:bg-black rounded-full blur-3xl pointer-events-none"
            />
            
            <div className="relative z-10 flex flex-col items-center text-center gap-6">
              <motion.div 
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3, type: 'spring', stiffness: 150, damping: 20 }}
                className="w-16 h-16 md:w-20 md:h-20 bg-[var(--color-off-black-ink)] dark:bg-white rounded-[var(--radius-3xl)] flex items-center justify-center mx-auto shadow-lg"
              >
                <ShieldAlert className="w-8 h-8 md:w-10 md:h-10 text-[var(--color-electric-lime)] dark:text-[var(--color-iris-gleam)]" />
              </motion.div>
              
              <div className="flex flex-col items-center">
                <motion.h2 
                  initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.4, duration: 0.6 }}
                  className="text-4xl md:text-5xl font-bold text-[var(--color-off-black-ink)] dark:text-white leading-tight tracking-[var(--tracking-heading)]"
                >
                  Payment<br/>Paused
                </motion.h2>
                <motion.p 
                  initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}
                  className="mt-4 text-[length:var(--text-body)] md:text-lg text-[var(--color-off-black-ink)] dark:text-white opacity-90 max-w-sm mx-auto"
                >
                  RakshaPay's on-device model intercepted this transaction due to a high probability of coercion or social engineering.
                </motion.p>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: The Analysis & Actions */}
          <div className="w-full md:w-7/12 flex flex-col h-full bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-obsidian)] relative transition-colors duration-300">
            <div className="flex-1 overflow-y-auto p-6 md:p-12 lg:px-20 lg:py-16">
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.6 }}
                className="mb-8 p-5 bg-[var(--color-pure-white)] dark:bg-[var(--color-graphite-dark)] border border-[var(--color-ash)] dark:border-[var(--color-steel)] rounded-[var(--radius-2xl)]"
              >
                <p className="text-[length:var(--text-body)] text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] leading-relaxed">
                  You are attempting to send <span className="font-bold text-xl">₹{amount}</span> to <br className="hidden md:block"/>
                  <span className="font-bold">{payee}</span> <span className="text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] text-sm">({upiId})</span>.
                </p>
                <p className="mt-3 text-[length:var(--text-body-sm)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)]">
                  We paused this because our system detected the following unusual environmental patterns happening simultaneously:
                </p>
              </motion.div>

              {/* Staggered Explainability List */}
              <motion.div 
                variants={containerVariants} 
                initial="hidden" 
                animate="show" 
                className="space-y-4"
              >
                {riskFactors.map((factor) => {
                  const Icon = factor.icon;
                  return (
                    <motion.div
                      variants={itemVariants}
                      key={factor.id}
                      className="bg-[var(--color-pure-white)] dark:bg-[var(--color-graphite-dark)] border border-[var(--color-ash)] dark:border-[var(--color-steel)] rounded-[var(--radius-lg)] p-5 flex items-start gap-4 shadow-sm"
                    >
                      <div className="p-3 bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-abyss)] rounded-full shrink-0">
                        <Icon className="w-6 h-6 text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)]" />
                      </div>
                      <div className="pt-0.5">
                        <h4 className="font-bold text-[length:var(--text-body)] text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)]">
                          {factor.title}
                        </h4>
                        <p className="text-[length:var(--text-body-sm)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] mt-1">
                          {factor.description}
                        </p>
                      </div>
                    </motion.div>
                  );
                })}
              </motion.div>
            </div>

            {/* Sticky Action Buttons at Bottom */}
            <motion.div 
              initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 1.2, duration: 0.6 }}
              className="p-6 md:px-12 lg:px-20 bg-gradient-to-t from-[var(--color-off-white-canvas)] dark:from-[var(--color-obsidian)] via-[var(--color-off-white-canvas)] dark:via-[var(--color-obsidian)] to-transparent shrink-0"
            >
              <div className="flex flex-col gap-3 max-w-2xl mx-auto">
                <button
                  type="button"
                  onClick={onCancel}
                  className="w-full py-4 px-6 rounded-[var(--radius-full)] bg-[var(--color-off-black-ink)] dark:bg-white text-[var(--color-electric-lime)] dark:text-[var(--color-obsidian)] font-bold text-[length:var(--text-body)] hover:scale-[1.02] transition-transform active:scale-[0.98] shadow-lg cursor-pointer"
                >
                  Cancel Payment (Recommended)
                </button>
                <button
                  type="button"
                  onClick={onProceed}
                  className="w-full py-4 px-6 rounded-[var(--radius-full)] bg-transparent border-2 border-[var(--color-ash)] dark:border-[var(--color-steel)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] hover:border-[var(--color-off-black-ink)] dark:hover:border-white hover:text-[var(--color-off-black-ink)] dark:hover:text-white font-bold text-[length:var(--text-body-sm)] transition-colors active:scale-[0.98] cursor-pointer"
                >
                  I understand the risks, send anyway
                </button>
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default InterventionModal;