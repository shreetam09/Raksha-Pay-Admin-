import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { Toaster, toast } from 'react-hot-toast';
import AuthScreen from './components/AuthScreen';
import PaymentScreen from './components/PaymentScreen';
import InterventionModal from './components/InterventionModal';

// We wrap the main content in a child component so we can use the `useNavigate` hook
function AppFlow() {
  const navigate = useNavigate();
  
  // Global Dark Mode State
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('rakshapay_theme') === 'dark';
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('rakshapay_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('rakshapay_theme', 'light');
    }
  }, [isDark]);

  // Centralized state for the Intervention Modal
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalData, setModalData] = useState({
    amount: '',
    upiId: '',
    telemetry: {}
  });

  // Handler for when the "Student Model" flags a transaction
  const handleTriggerIntervention = (data) => {
    setModalData(data);
    setIsModalOpen(true);
  };

  // Handler for a safe transaction (no coercion detected)
  const handlePaymentSuccess = (data) => {
    // In a real app, this would route to a success screen
    toast.success('Safe Transaction! Money sent successfully.', {
      style: {
        background: 'var(--color-off-black-ink)',
        color: 'var(--color-pure-white)',
      },
      iconTheme: {
        primary: 'var(--color-electric-lime)',
        secondary: 'black',
      },
    });
  };

  return (
    <div className="bg-[var(--color-pure-white)] dark:bg-[var(--color-obsidian)] text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)] min-h-screen relative font-sans transition-colors duration-300">
      <Toaster position="top-center" />
      <Routes>
        {/* Route 1: The Login/Onboarding Screen */}
        <Route 
          path="/" 
          element={
            <AuthScreen 
              onLogin={() => navigate('/app')} 
              isDark={isDark}
              setIsDark={setIsDark}
            />
          } 
        />
        
        {/* Route 2: The Consumer Banking App & Dev Terminal */}
        <Route 
          path="/app" 
          element={
            <PaymentScreen 
              onTriggerIntervention={handleTriggerIntervention}
              onPaymentSuccess={handlePaymentSuccess}
              isDark={isDark}
              setIsDark={setIsDark}
            />
          } 
        />
      </Routes>

      {/* The Shield Intervention Modal (Sits on top of the entire app) */}
      <InterventionModal 
        isOpen={isModalOpen}
        onCancel={() => setIsModalOpen(false)}
        onProceed={() => {
          setIsModalOpen(false);
          toast.error('User Overrode Shield: Transaction forced through.', {
            style: {
              background: '#ff4433',
              color: '#fff',
            },
          });
        }}
        amount={modalData.amount}
        payee="Unknown" // You can pass actual payee names here if you have a contact book lookup
        upiId={modalData.upiId}
      />
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppFlow />
    </Router>
  );
}
